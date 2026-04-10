use std::{
    cmp::Ordering,
    path::{Path, PathBuf},
};

pub(crate) fn sort_paths_natural(paths: &mut [PathBuf]) {
    paths.sort_by(|lhs, rhs| {
        let lhs_name = path_sort_key(lhs);
        let rhs_name = path_sort_key(rhs);
        compare_natural(lhs_name.as_str(), rhs_name.as_str()).then_with(|| lhs.cmp(rhs))
    });
}

fn path_sort_key(path: &Path) -> String {
    path.file_name()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| path.to_string_lossy().to_ascii_lowercase())
}

fn compare_natural(lhs: &str, rhs: &str) -> Ordering {
    let lhs = lhs.as_bytes();
    let rhs = rhs.as_bytes();
    let mut lhs_idx = 0usize;
    let mut rhs_idx = 0usize;

    while lhs_idx < lhs.len() && rhs_idx < rhs.len() {
        let lhs_byte = lhs[lhs_idx];
        let rhs_byte = rhs[rhs_idx];

        if lhs_byte.is_ascii_digit() && rhs_byte.is_ascii_digit() {
            let lhs_start = lhs_idx;
            let rhs_start = rhs_idx;

            while lhs_idx < lhs.len() && lhs[lhs_idx].is_ascii_digit() {
                lhs_idx += 1;
            }
            while rhs_idx < rhs.len() && rhs[rhs_idx].is_ascii_digit() {
                rhs_idx += 1;
            }

            let lhs_digits = &lhs[lhs_start..lhs_idx];
            let rhs_digits = &rhs[rhs_start..rhs_idx];
            let lhs_trimmed = trim_leading_zeros(lhs_digits);
            let rhs_trimmed = trim_leading_zeros(rhs_digits);

            match lhs_trimmed.len().cmp(&rhs_trimmed.len()) {
                Ordering::Equal => {}
                ordering => return ordering,
            }

            match lhs_trimmed.cmp(rhs_trimmed) {
                Ordering::Equal => {}
                ordering => return ordering,
            }

            match lhs_digits.len().cmp(&rhs_digits.len()) {
                Ordering::Equal => {}
                ordering => return ordering,
            }

            continue;
        }

        match lhs_byte.cmp(&rhs_byte) {
            Ordering::Equal => {
                lhs_idx += 1;
                rhs_idx += 1;
            }
            ordering => return ordering,
        }
    }

    lhs.len().cmp(&rhs.len())
}

fn trim_leading_zeros(digits: &[u8]) -> &[u8] {
    let mut idx = 0usize;
    while idx + 1 < digits.len() && digits[idx] == b'0' {
        idx += 1;
    }
    &digits[idx..]
}

#[cfg(test)]
mod tests {
    use super::sort_paths_natural;
    use std::path::PathBuf;

    #[test]
    fn sorts_numeric_suffixes_in_human_order() {
        let mut paths = vec![
            PathBuf::from("10.png"),
            PathBuf::from("2.png"),
            PathBuf::from("1.png"),
            PathBuf::from("frame_11.png"),
            PathBuf::from("frame_3.png"),
        ];

        sort_paths_natural(&mut paths);

        let actual = paths
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            vec![
                "1.png".to_string(),
                "2.png".to_string(),
                "10.png".to_string(),
                "frame_3.png".to_string(),
                "frame_11.png".to_string(),
            ]
        );
    }
}
