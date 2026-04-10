use std::{
    fs::{File, OpenOptions},
    io,
    path::PathBuf,
    sync::{Mutex, MutexGuard, OnceLock},
};

use fs2::FileExt;

fn process_lock() -> &'static Mutex<()> {
    static PROCESS_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    PROCESS_LOCK.get_or_init(|| Mutex::new(()))
}

fn lock_path() -> PathBuf {
    std::env::temp_dir().join("ferrismind-lux3d-gpu-tests.lock")
}

#[derive(Debug)]
pub struct GpuTestLock {
    _process_guard: MutexGuard<'static, ()>,
    file: File,
}

impl GpuTestLock {
    pub fn acquire() -> io::Result<Self> {
        let process_guard = process_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(lock_path())?;
        file.lock_exclusive()?;
        Ok(Self {
            _process_guard: process_guard,
            file,
        })
    }
}

impl Drop for GpuTestLock {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self.file);
    }
}
