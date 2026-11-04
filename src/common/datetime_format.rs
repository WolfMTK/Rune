const MICROS_PER_SECOND: i128 = 1_000_000;
const MICROS_PER_MINUTE: i128 = 60 * MICROS_PER_SECOND;
const MICROS_PER_HOUR: i128 = 60 * MICROS_PER_MINUTE;
const MICROS_PER_DAY: i128 = 24 * MICROS_PER_HOUR;
const MINUTES_PER_DAY: i128 = 24 * 60;
const MAX_DURATION_DAYS: i128 = 999_999_999;
const MAX_DURATION_MICROS: i128 = MAX_DURATION_DAYS * MICROS_PER_DAY;
const MAX_DURATION_FRACTION_DIGITS: usize = 11;

pub(crate) struct Date {
    pub(crate) year: i32,
    pub(crate) month: u8,
    pub(crate) day: u8,
}

pub(crate) struct Time {
    pub(crate) hour: u8,
    pub(crate) minute: u8,
    pub(crate) second: u8,
    pub(crate) microsecond: u32,
    pub(crate) offset_minutes: Option<i32>,
}

pub(crate) enum DurationError {
    Malformed,
    UnsupportedUnit,
    OutOfRange,
}

pub(crate) fn parse_date(text: &str) -> Option<Date> {
    let bytes = text.as_bytes();
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return None;
    }
    let year = i32::try_from(digits_value(&bytes[0..4])?).ok()?;
    let month = u8::try_from(digits_value(&bytes[5..7])?).ok()?;
    let day = u8::try_from(digits_value(&bytes[8..10])?).ok()?;
    let is_valid = (1..=9999).contains(&year)
        && (1..=12).contains(&month)
        && (1..=days_in_month(year, month)).contains(&day);
    is_valid.then_some(Date { year, month, day })
}

pub(crate) struct DateTime {
    pub(crate) date: Date,
    pub(crate) time: Time,
}

pub(crate) fn parse_datetime(text: &str) -> Option<DateTime> {
    if !matches!(text.as_bytes().get(10), Some(b'T' | b't' | b' ')) {
        return None;
    }
    let date = parse_date(text.get(..10)?)?;
    let (time, next_day) = parse_clock(text.get(11..)?)?;
    let date = if next_day { next_date(date)? } else { date };
    Some(DateTime { date, time })
}

pub(crate) fn parse_time(text: &str) -> Option<Time> {
    parse_clock(text).map(|(time, _)| time)
}

fn next_date(date: Date) -> Option<Date> {
    let Date { year, month, day } = date;
    if day < days_in_month(year, month) {
        Some(Date {
            year,
            month,
            day: day + 1,
        })
    } else if month < 12 {
        Some(Date {
            year,
            month: month + 1,
            day: 1,
        })
    } else if year < 9999 {
        Some(Date {
            year: year + 1,
            month: 1,
            day: 1,
        })
    } else {
        None
    }
}

fn parse_clock(text: &str) -> Option<(Time, bool)> {
    let bytes = text.as_bytes();
    if bytes.len() < 8 || bytes[2] != b':' || bytes[5] != b':' {
        return None;
    }
    let hour = digits_value(&bytes[0..2])?;
    let minute = digits_value(&bytes[3..5])?;
    let second = digits_value(&bytes[6..8])?;
    if hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    let (fraction, suffix) = split_fraction(&bytes[8..])?;
    let offset_minutes = parse_suffix(suffix)?;
    let micros_of_day = ((u64::from(hour) * 60 + u64::from(minute)) * 60 + u64::from(second))
        * 1_000_000
        + u64::from(fraction_micros(fraction));
    let day_micros = 24 * 3_600 * 1_000_000;
    let next_day = micros_of_day >= day_micros;
    let micros_of_day = micros_of_day % day_micros;
    let time = Time {
        hour: (micros_of_day / 3_600_000_000) as u8,
        minute: ((micros_of_day / 60_000_000) % 60) as u8,
        second: ((micros_of_day / 1_000_000) % 60) as u8,
        microsecond: (micros_of_day % 1_000_000) as u32,
        offset_minutes,
    };
    Some((time, next_day))
}

pub(crate) fn parse_duration(text: &str) -> Result<i128, DurationError> {
    let bytes = text.as_bytes();
    let (negative, body) = match bytes.first() {
        Some(b'-') => (true, &bytes[1..]),
        Some(b'+') => (false, &bytes[1..]),
        _ => (false, bytes),
    };
    let Some((b'P' | b'p', rest)) = body.split_first() else {
        return Err(DurationError::Malformed);
    };

    let mut total = Some(0_i128);
    let mut in_time = false;
    let mut last_rank = 0_u8;
    let mut has_fraction = false;
    let mut has_component = false;
    let mut position = 0;
    while position < rest.len() {
        if rest[position].eq_ignore_ascii_case(&b'T') {
            if in_time {
                return Err(DurationError::Malformed);
            }
            in_time = true;
            position += 1;
            if position == rest.len() {
                return Err(DurationError::Malformed);
            }
            continue;
        }
        if has_fraction {
            return Err(DurationError::Malformed);
        }
        let integer_start = position;
        position += count_digits(&rest[position..]);
        let integer = &rest[integer_start..position];
        if integer.is_empty() {
            return Err(DurationError::Malformed);
        }
        let mut fraction: &[u8] = &[];
        if rest.get(position) == Some(&b'.') {
            let fraction_start = position + 1;
            let fraction_len = count_digits(&rest[fraction_start..]);
            if fraction_len == 0 {
                return Err(DurationError::Malformed);
            }
            fraction = &rest[fraction_start..fraction_start + fraction_len];
            position = fraction_start + fraction_len;
            has_fraction = true;
        }
        let Some(designator) = rest.get(position) else {
            return Err(DurationError::Malformed);
        };
        position += 1;
        let (rank, unit_micros) = match (in_time, designator.to_ascii_uppercase()) {
            (false, b'D') => (1, MICROS_PER_DAY),
            (false, b'Y' | b'M' | b'W') => return Err(DurationError::UnsupportedUnit),
            (true, b'H') => (2, MICROS_PER_HOUR),
            (true, b'M') => (3, MICROS_PER_MINUTE),
            (true, b'S') => (4, MICROS_PER_SECOND),
            _ => return Err(DurationError::Malformed),
        };
        if rank <= last_rank {
            return Err(DurationError::Malformed);
        }
        last_rank = rank;
        has_component = true;
        let component = component_micros(integer, fraction, unit_micros);
        total = match (total, component) {
            (Some(sum), Some(micros)) => sum.checked_add(micros),
            _ => None,
        };
    }
    if !has_component {
        return Err(DurationError::Malformed);
    }

    let magnitude = total.ok_or(DurationError::OutOfRange)?;
    if negative {
        if magnitude > MAX_DURATION_MICROS {
            return Err(DurationError::OutOfRange);
        }
        Ok(-magnitude)
    } else if magnitude > MAX_DURATION_MICROS + MICROS_PER_DAY - 1 {
        Err(DurationError::OutOfRange)
    } else {
        Ok(magnitude)
    }
}

pub(crate) fn format_date(year: i32, month: u8, day: u8) -> String {
    format!("{year:04}-{month:02}-{day:02}")
}

pub(crate) fn format_time(
    hour: u8,
    minute: u8,
    second: u8,
    microsecond: u32,
    offset_minutes: Option<i32>,
) -> String {
    let fraction = if microsecond == 0 {
        String::new()
    } else {
        format!(".{microsecond:06}")
    };
    let offset = match offset_minutes {
        None => String::new(),
        Some(0) => "Z".to_owned(),
        Some(minutes) => format_offset(minutes),
    };
    format!("{hour:02}:{minute:02}:{second:02}{fraction}{offset}")
}

pub(crate) fn format_duration(total_micros: i128) -> String {
    let sign = if total_micros < 0 { "-" } else { "" };
    let magnitude = total_micros.unsigned_abs();
    let day_micros = MICROS_PER_DAY.unsigned_abs();
    let days = magnitude / day_micros;
    let rest = magnitude % day_micros;
    let seconds = rest / 1_000_000;
    let microseconds = rest % 1_000_000;
    let has_time = seconds != 0 || microseconds != 0;
    let days_text = if days > 0 || !has_time {
        format!("{days}D")
    } else {
        String::new()
    };
    let time_text = if has_time {
        format!("T{seconds}{}S", fraction_text(microseconds))
    } else {
        String::new()
    };
    format!("{sign}P{days_text}{time_text}")
}

pub(crate) fn utc_offset_minutes(offset_micros: i128) -> i32 {
    let whole_minutes = offset_micros / MICROS_PER_MINUTE;
    let remainder = offset_micros % MICROS_PER_MINUTE;
    let rounded = if 2 * remainder.abs() > MICROS_PER_MINUTE {
        whole_minutes + remainder.signum()
    } else {
        whole_minutes
    };
    (rounded % MINUTES_PER_DAY) as i32
}

pub(crate) fn total_micros(days: i32, seconds: i32, microseconds: i32) -> i128 {
    i128::from(days) * MICROS_PER_DAY
        + i128::from(seconds) * MICROS_PER_SECOND
        + i128::from(microseconds)
}

pub(crate) fn timedelta_components(total_micros: i128) -> (i32, i32, i32) {
    let days = total_micros.div_euclid(MICROS_PER_DAY);
    let rest = total_micros.rem_euclid(MICROS_PER_DAY);
    (
        days as i32,
        (rest / MICROS_PER_SECOND) as i32,
        (rest % MICROS_PER_SECOND) as i32,
    )
}

fn format_offset(minutes: i32) -> String {
    let sign = if minutes < 0 { '-' } else { '+' };
    let magnitude = minutes.unsigned_abs();
    format!("{sign}{:02}:{:02}", magnitude / 60, magnitude % 60)
}

fn fraction_text(microseconds: u128) -> String {
    if microseconds == 0 {
        return String::new();
    }
    let digits = format!("{microseconds:06}");
    format!(".{}", digits.trim_end_matches('0'))
}

fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        2 if is_leap_year(year) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

fn is_leap_year(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn digits_value(bytes: &[u8]) -> Option<u32> {
    bytes.iter().try_fold(0_u32, |value, byte| {
        byte.is_ascii_digit()
            .then(|| value * 10 + u32::from(byte - b'0'))
    })
}

fn digits_i128(bytes: &[u8]) -> Option<i128> {
    bytes.iter().try_fold(0_i128, |value, byte| {
        value.checked_mul(10)?.checked_add(i128::from(byte - b'0'))
    })
}

fn count_digits(bytes: &[u8]) -> usize {
    bytes
        .iter()
        .take_while(|byte| byte.is_ascii_digit())
        .count()
}

fn split_fraction(rest: &[u8]) -> Option<(&[u8], &[u8])> {
    match rest.strip_prefix(b".") {
        Some(after_point) => {
            let digit_count = count_digits(after_point);
            (digit_count > 0).then(|| after_point.split_at(digit_count))
        }
        None => Some((&rest[..0], rest)),
    }
}

fn fraction_micros(digits: &[u8]) -> u32 {
    let mut micros = 0;
    for position in 0..6 {
        let digit = digits
            .get(position)
            .map_or(0, |digit| u32::from(digit - b'0'));
        micros = micros * 10 + digit;
    }
    let round_up = digits.get(6).is_some_and(|digit| *digit >= b'5');
    micros + u32::from(round_up)
}

fn parse_suffix(suffix: &[u8]) -> Option<Option<i32>> {
    match suffix {
        [] => Some(None),
        [b'Z' | b'z'] => Some(Some(0)),
        [sign @ (b'+' | b'-'), offset @ ..] => parse_offset(*sign == b'-', offset).map(Some),
        _ => None,
    }
}

fn parse_offset(negative: bool, bytes: &[u8]) -> Option<i32> {
    let (hours, minutes) = match bytes {
        [h1, h2, b':', m1, m2] => (digits_value(&[*h1, *h2])?, digits_value(&[*m1, *m2])?),
        [h1, h2, m1, m2] => (digits_value(&[*h1, *h2])?, digits_value(&[*m1, *m2])?),
        _ => return None,
    };
    if hours > 23 || minutes > 59 {
        return None;
    }
    let total = i32::try_from(hours * 60 + minutes).ok()?;
    Some(if negative { -total } else { total })
}

fn component_micros(integer: &[u8], fraction: &[u8], unit_micros: i128) -> Option<i128> {
    let whole = digits_i128(integer)?.checked_mul(unit_micros)?;
    let kept = &fraction[..fraction.len().min(MAX_DURATION_FRACTION_DIGITS)];
    let numerator = digits_i128(kept)?.checked_mul(unit_micros)?;
    let denominator = 10_i128.pow(u32::try_from(kept.len()).ok()?);
    let rounded = (2 * numerator + denominator) / (2 * denominator);
    whole.checked_add(rounded)
}
