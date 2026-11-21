#![allow(clippy::undocumented_unsafe_blocks)]

use std::ptr;

use pyo3::{
    Bound, Py, PyAny, PyResult, Python, ffi, intern,
    prelude::PyDictMethods,
    sync::PyOnceLock,
    types::{
        PyAnyMethods, PyDate, PyDateAccess, PyDateTime, PyDelta, PyDeltaAccess, PyDict, PyFloat,
        PyFloatMethods, PyTime, PyTimeAccess, PyTzInfo,
    },
};

use crate::{
    common::{
        py_typing::{date, datetime, time, timedelta},
        structures::{
            datetime_format,
            errors::{Fail, overflow_failure},
        },
        types::{Decoded, Outcome},
    },
    constants::{MICROS_PER_DAY, MICROS_PER_HOUR, MICROS_PER_MINUTE, MICROS_PER_SECOND},
};

pub(crate) fn timedelta_from_seconds<'py>(
    py: Python<'py>,
    seconds: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "seconds"), seconds)?;
    timedelta(py)?.call((), Some(&kwargs))
}

pub(crate) fn duration_from_seconds<'py>(
    py: Python<'py>,
    seconds: &Bound<'py, PyAny>,
) -> Decoded<'py> {
    timedelta_from_seconds(py, seconds)
        .map_err(|err| overflow_failure(py, err, "Duration is out of range"))
}

pub(crate) fn datetime_from_epoch<'py>(
    py: Python<'py>,
    seconds: &Bound<'py, PyAny>,
) -> Decoded<'py> {
    if seconds
        .cast::<PyFloat>()
        .is_ok_and(|value| !value.value().is_finite())
    {
        return Err(Fail::invalid("Invalid epoch timestamp".to_owned()));
    }
    let utc = PyTzInfo::utc(py)?;
    let epoch = PyDateTime::new(py, 1970, 1, 1, 0, 0, 0, 0, Some(&utc))?;
    timedelta_from_seconds(py, seconds)
        .and_then(|offset| epoch.add(offset))
        .map_err(|err| overflow_failure(py, err, "Epoch timestamp is out of range"))
}

pub(crate) fn datetime_from_timestamp<'py>(py: Python<'py>, data: &[u8]) -> Decoded<'py> {
    let invalid = || Fail::invalid("Invalid MessagePack timestamp".to_owned());
    let (seconds, nanos) = match data.len() {
        4 => (
            i64::from(u32::from_be_bytes(data.try_into().map_err(|_| invalid())?)),
            0,
        ),
        8 => {
            let packed = u64::from_be_bytes(data.try_into().map_err(|_| invalid())?);
            ((packed & 0x3_ffff_ffff) as i64, (packed >> 34) as u32)
        }
        12 => (
            i64::from_be_bytes(data[4..].try_into().map_err(|_| invalid())?),
            u32::from_be_bytes(data[..4].try_into().map_err(|_| invalid())?),
        ),
        _ => return Err(invalid()),
    };
    if nanos > 999_999_999 {
        return Err(invalid());
    }
    let micros = i128::from(seconds) * MICROS_PER_SECOND + i128::from((nanos + 500) / 1000);
    let (days, seconds, microseconds) = datetime_format::timedelta_components(micros);
    let utc = PyTzInfo::utc(py)?;
    let epoch = PyDateTime::new(py, 1970, 1, 1, 0, 0, 0, 0, Some(&utc))?;
    PyDelta::new(py, days, seconds, microseconds, false)
        .and_then(|offset| epoch.add(offset))
        .map_err(|err| overflow_failure(py, err, "Timestamp is out of range"))
}

pub(crate) fn timestamp_payload(seconds: i64, nanos: u32) -> Vec<u8> {
    if seconds >> 34 == 0 {
        let packed = (u64::from(nanos) << 34) | seconds.cast_unsigned();
        if packed >> 32 == 0 {
            return (packed as u32).to_be_bytes().to_vec();
        }
        return packed.to_be_bytes().to_vec();
    }
    let mut payload = nanos.to_be_bytes().to_vec();
    payload.extend_from_slice(&seconds.to_be_bytes());
    payload
}

pub(crate) fn date_from_text<'py>(py: Python<'py>, text: &str) -> Outcome<Bound<'py, PyAny>> {
    let parsed = datetime_format::parse_date(text)
        .ok_or_else(|| Fail::invalid("Invalid RFC3339 date".to_owned()))?;
    Ok(PyDate::new(py, parsed.year, parsed.month, parsed.day)?.into_any())
}

pub(crate) fn time_from_text<'py>(py: Python<'py>, text: &str) -> Outcome<Bound<'py, PyAny>> {
    let parsed = datetime_format::parse_time(text)
        .ok_or_else(|| Fail::invalid("Invalid RFC3339 time".to_owned()))?;
    let tzinfo = match parsed.offset_minutes {
        Some(minutes) => Some(fixed_timezone(py, minutes)?),
        None => None,
    };
    let value = PyTime::new(
        py,
        parsed.hour,
        parsed.minute,
        parsed.second,
        parsed.microsecond,
        tzinfo.as_ref(),
    )?;
    Ok(value.into_any())
}

pub(crate) fn datetime_from_text<'py>(py: Python<'py>, text: &str) -> Outcome<Bound<'py, PyAny>> {
    let parsed = datetime_format::parse_datetime(text)
        .ok_or_else(|| Fail::invalid("Invalid RFC3339 datetime".to_owned()))?;
    let time = parsed.time;
    let tzinfo = match time.offset_minutes {
        Some(minutes) => Some(fixed_timezone(py, minutes)?),
        None => None,
    };
    let value = PyDateTime::new(
        py,
        parsed.date.year,
        parsed.date.month,
        parsed.date.day,
        time.hour,
        time.minute,
        time.second,
        time.microsecond,
        tzinfo.as_ref(),
    )?;
    Ok(value.into_any())
}

pub(crate) fn timedelta_from_text<'py>(py: Python<'py>, text: &str) -> Outcome<Bound<'py, PyAny>> {
    let total_micros = datetime_format::parse_duration(text).map_err(duration_failure)?;
    let (days, seconds, microseconds) = datetime_format::timedelta_components(total_micros);
    Ok(PyDelta::new(py, days, seconds, microseconds, false)?.into_any())
}

pub(crate) fn fixed_timezone<'py>(py: Python<'py>, minutes: i32) -> PyResult<Bound<'py, PyTzInfo>> {
    if minutes == 0 {
        return PyTzInfo::utc(py).map(|value| value.to_owned());
    }
    static CACHE: PyOnceLock<Py<PyDict>> = PyOnceLock::new();
    let cache = CACHE.get_or_init(py, || PyDict::new(py).unbind()).bind(py);
    if let Some(tzinfo) = cache.get_item(minutes)? {
        return Ok(tzinfo.cast_into::<PyTzInfo>()?);
    }
    let tzinfo = PyTzInfo::fixed_offset(py, PyDelta::new(py, 0, minutes * 60, 0, true)?)?;
    cache.set_item(minutes, &tzinfo)?;
    Ok(tzinfo)
}

pub(crate) fn temporal_text(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    let mut storage = [0_u8; 40];
    let mut buf = datetime_format::TextBuf::new(&mut storage);
    temporal_text_into(obj, &mut buf)?;
    Ok(buf.as_str().to_owned())
}

#[inline]
pub(crate) fn temporal_text_into(
    obj: &Bound<'_, PyAny>,
    buf: &mut datetime_format::TextBuf<'_>,
) -> PyResult<()> {
    let py = obj.py();
    let ty = obj.get_type();
    if ty.is(datetime(py)?) {
        datetime_text_into(obj, buf)
    } else if ty.is(date(py)?) {
        date_text_into(obj, buf)
    } else if ty.is(time(py)?) {
        time_text_into(obj, buf)
    } else {
        timedelta_text_into(obj, buf)
    }
}

pub(crate) fn datetime_text_into(
    obj: &Bound<'_, PyAny>,
    buf: &mut datetime_format::TextBuf<'_>,
) -> PyResult<()> {
    let value = obj.cast::<PyDateTime>()?;
    datetime_format::write_date_text(buf, value.get_year(), value.get_month(), value.get_day());
    buf.push(b'T');
    datetime_format::write_time_text(
        buf,
        value.get_hour(),
        value.get_minute(),
        value.get_second(),
        value.get_microsecond(),
        utcoffset_minutes(obj, DATETIME_TZINFO_OFFSET)?,
    );
    Ok(())
}

pub(crate) fn date_text_into(
    obj: &Bound<'_, PyAny>,
    buf: &mut datetime_format::TextBuf<'_>,
) -> PyResult<()> {
    let value = obj.cast::<PyDate>()?;
    datetime_format::write_date_text(buf, value.get_year(), value.get_month(), value.get_day());
    Ok(())
}

pub(crate) fn time_text_into(
    obj: &Bound<'_, PyAny>,
    buf: &mut datetime_format::TextBuf<'_>,
) -> PyResult<()> {
    let value = obj.cast::<PyTime>()?;
    datetime_format::write_time_text(
        buf,
        value.get_hour(),
        value.get_minute(),
        value.get_second(),
        value.get_microsecond(),
        utcoffset_minutes(obj, TIME_TZINFO_OFFSET)?,
    );
    Ok(())
}

pub(crate) fn timedelta_text_into(
    obj: &Bound<'_, PyAny>,
    buf: &mut datetime_format::TextBuf<'_>,
) -> PyResult<()> {
    let value = obj.cast::<PyDelta>()?;
    datetime_format::write_duration_text(
        buf,
        datetime_format::total_micros(
            value.get_days(),
            value.get_seconds(),
            value.get_microseconds(),
        ),
    );
    Ok(())
}

pub(crate) fn epoch_micros(obj: &Bound<'_, PyAny>) -> PyResult<Option<i128>> {
    let Ok(value) = obj.cast_exact::<PyDateTime>() else {
        return Ok(None);
    };
    let Some(offset_micros) = utcoffset_micros(obj, DATETIME_TZINFO_OFFSET)? else {
        return Ok(None);
    };
    let days =
        datetime_format::days_from_civil(value.get_year(), value.get_month(), value.get_day());
    Ok(Some(
        i128::from(days) * MICROS_PER_DAY
            + i128::from(value.get_hour()) * MICROS_PER_HOUR
            + i128::from(value.get_minute()) * MICROS_PER_MINUTE
            + i128::from(value.get_second()) * MICROS_PER_SECOND
            + i128::from(value.get_microsecond())
            - offset_micros,
    ))
}

pub(crate) fn utcoffset_minutes(
    obj: &Bound<'_, PyAny>,
    tzinfo_offset: usize,
) -> PyResult<Option<i32>> {
    let py = obj.py();
    let tzinfo = unsafe { temporal_tzinfo(obj, tzinfo_offset) };
    if tzinfo.is_null() || ptr::eq(tzinfo, unsafe { ffi::Py_None() }) {
        return Ok(None);
    }
    if ptr::eq(tzinfo, timezone_utc(py)?.as_ptr()) {
        return Ok(Some(0));
    }
    let offset = obj.call_method0(intern!(py, "utcoffset"))?;
    if offset.is_none() {
        return Ok(None);
    }
    let delta = offset.cast::<PyDelta>()?;
    Ok(Some(datetime_format::utc_offset_minutes(
        datetime_format::total_micros(
            delta.get_days(),
            delta.get_seconds(),
            delta.get_microseconds(),
        ),
    )))
}

static TIMEZONE_UTC: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

fn timezone_utc(py: Python<'_>) -> PyResult<&Py<PyAny>> {
    TIMEZONE_UTC.get_or_try_init(py, || {
        py.import("datetime")?
            .getattr("timezone")?
            .getattr("utc")
            .map(Bound::unbind)
    })
}

const TEMPORAL_HASTZINFO_OFFSET: usize = 24;
const DATETIME_TZINFO_OFFSET: usize = 40;
const TIME_TZINFO_OFFSET: usize = 32;

#[inline(always)]
unsafe fn temporal_tzinfo(obj: &Bound<'_, PyAny>, tzinfo_offset: usize) -> *mut ffi::PyObject {
    let has_tzinfo = unsafe { *obj.as_ptr().cast::<u8>().add(TEMPORAL_HASTZINFO_OFFSET) } != 0;
    if !has_tzinfo {
        return ptr::null_mut();
    }
    unsafe { *obj.as_ptr().cast::<u8>().add(tzinfo_offset).cast() }
}

#[inline]
fn utcoffset_micros(obj: &Bound<'_, PyAny>, tzinfo_offset: usize) -> PyResult<Option<i128>> {
    let py = obj.py();
    let tzinfo = unsafe { temporal_tzinfo(obj, tzinfo_offset) };
    if tzinfo.is_null() {
        return Ok(None);
    }
    if ptr::eq(tzinfo, py.None().as_ptr()) {
        return Ok(None);
    }
    if ptr::eq(tzinfo, timezone_utc(py)?.as_ptr()) {
        return Ok(Some(0));
    }
    let offset = obj.call_method0(intern!(py, "utcoffset"))?;
    if offset.is_none() {
        return Ok(None);
    }
    let delta = offset.cast::<PyDelta>()?;
    Ok(Some(datetime_format::total_micros(
        delta.get_days(),
        delta.get_seconds(),
        delta.get_microseconds(),
    )))
}

fn duration_failure(error: datetime_format::DurationError) -> Fail {
    Fail::invalid(match error {
        datetime_format::DurationError::Malformed => "Invalid ISO8601 duration".to_owned(),
        datetime_format::DurationError::UnsupportedUnit => {
            "Only units 'D', 'H', 'M', and 'S' are supported in durations".to_owned()
        }
        datetime_format::DurationError::OutOfRange => "Duration is out of range".to_owned(),
    })
}
