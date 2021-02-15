use std::cell::RefCell;
use std::mem;

pub struct Arena<T> {
    slabs: RefCell<Slabs<T>>,
}

struct Slabs<T> {
    current: Vec<T>,
    previous: Vec<Vec<T>>,
}

impl<T> Arena<T> {
    const MIN_SLAB_SIZE: usize = 1024;

    pub fn new() -> Arena<T> {
        Arena {
            slabs: RefCell::new(Slabs {
                current: vec![],
                previous: vec![],
            }),
        }
    }

    pub fn alloc(&self, value: T) -> &mut T {
        let mut slabs = self.slabs.borrow_mut();

        if slabs.current.len() >= slabs.current.capacity() {
            let capacity = if slabs.current.is_empty() {
                let size_of = usize::max(1, mem::size_of::<T>());
                usize::max(Self::MIN_SLAB_SIZE / size_of, 1)
            } else {
                2 * slabs.current.len()
            };

            let mut slab = Vec::with_capacity(capacity);
            mem::swap(&mut slab, &mut slabs.current);

            if !slab.is_empty() {
                slabs.previous.push(slab);
            }
        }

        let i = slabs.current.len();
        debug_assert!(i < slabs.current.capacity());
        slabs.current.push(value);

        unsafe { &mut *slabs.current.as_mut_ptr().add(i) }
    }
}
