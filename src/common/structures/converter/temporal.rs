use pyo3::{
    Bound, PyAny, Python, intern,
    prelude::PyAnyMethods,
    types::{PyDict, PyDictMethods},
};
use uuid::Uuid;

use crate::{
    common::{
        enums::Scalar,
        lax,
        py_typing::uuid,
        reader::{Reader, kind},
        structures::{
            converter::decode::Converter,
            errors::Fail,
            temporal::{
                date_from_text, datetime_from_epoch, datetime_from_text, datetime_from_timestamp,
                duration_from_seconds, time_from_text, timedelta_from_text,
            },
        },
        types::{Decoded, Outcome},
    },
    constants::TIMESTAMP_EXT_CODE,
};

impl<'p, 'py, 'a, W: Reader<'a>> Converter<'p, 'py, W> {
    pub(crate) fn decode_date(&mut self) -> Decoded<'py> {
        self.read_temporal_text("date", date_from_text)
    }
    pub(crate) fn decode_time(&mut self) -> Decoded<'py> {
        self.read_temporal_text("time", time_from_text)
    }
    pub(crate) fn decode_timedelta(&mut self, accepted: u16) -> Decoded<'py> {
        let py = self.py;
        self.expect_kind(accepted, "duration", kind::container_name)?;
        let scalar = self.wire.read_scalar()?;
        if self.plan.lax
            && let Some(seconds) = lax::numeric(py, &scalar)?
        {
            return duration_from_seconds(py, &seconds);
        }
        let Scalar::Str(text) = scalar else {
            unreachable!("kind is checked as str above")
        };
        timedelta_from_text(py, &text)
    }
    pub(crate) fn decode_datetime(&mut self, accepted: u16) -> Decoded<'py> {
        let py = self.py;
        self.expect_kind(accepted, "datetime", kind::name)?;
        let scalar = self.wire.read_scalar()?;
        if let Scalar::Ext { code, data } = scalar {
            if code != TIMESTAMP_EXT_CODE {
                return Err(Fail::mismatch("datetime", kind::name(kind::EXT)));
            }
            return datetime_from_timestamp(py, data);
        }
        if self.plan.lax
            && let Some(seconds) = lax::numeric(py, &scalar)?
        {
            return datetime_from_epoch(py, &seconds);
        }
        let Scalar::Str(text) = scalar else {
            unreachable!("kind is checked as str above")
        };
        datetime_from_text(py, &text)
    }
    pub(crate) fn decode_uuid(&mut self) -> Decoded<'py> {
        let py = self.py;
        let found = self.expect_kind(kind::STR | kind::BYTES, "UUID", kind::name)?;
        if found & kind::STR != 0 {
            let Scalar::Str(text) = self.wire.read_scalar()? else {
                unreachable!("kind is checked as str above")
            };
            let value =
                Uuid::try_parse(&text).map_err(|_| Fail::invalid("Invalid UUID".to_owned()))?;
            let kwargs = PyDict::new(py);
            kwargs.set_item(intern!(py, "int"), value.as_u128())?;
            return Ok(uuid(py)?.call((), Some(&kwargs))?);
        }
        let Scalar::Bytes(raw) = self.wire.read_scalar()? else {
            unreachable!("kind is checked as bytes above")
        };
        let bytes = <[u8; 16]>::try_from(raw.as_ref())
            .map_err(|_| Fail::invalid("Invalid UUID bytes".to_owned()))?;
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "int"), u128::from_be_bytes(bytes))?;
        Ok(uuid(py)?.call((), Some(&kwargs))?)
    }
    fn read_temporal_text(
        &mut self,
        expected: &str,
        decode: fn(Python<'py>, &str) -> Outcome<Bound<'py, PyAny>>,
    ) -> Decoded<'py> {
        self.expect_kind(kind::STR, expected, kind::container_name)?;
        let Scalar::Str(text) = self.wire.read_scalar()? else {
            unreachable!("kind is checked as str above")
        };
        decode(self.py, &text)
    }
}
