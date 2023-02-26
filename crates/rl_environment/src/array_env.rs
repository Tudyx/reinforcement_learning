use crate::{Gym3Environment, Step};

pub trait ArrayEnvironment<const N: usize> {
    type Action;
    type Observation;

    fn act(&mut self, actions: [Self::Action; N]);

    fn observe(&self) -> [Step<Self::Observation>; N];
}

/// Produce observation on array, no heap allocation here.
pub struct ArrayEnv<E, const N: usize>
where
    E: Gym3Environment,
{
    envs: [E; N],
}

impl<E, const N: usize> ArrayEnvironment<N> for ArrayEnv<E, N>
where
    E: Gym3Environment,
{
    type Action = E::Action;

    type Observation = E::Observation;

    fn act(&mut self, actions: [Self::Action; N]) {
        for (action, env) in actions.into_iter().zip(self.envs.iter_mut()) {
            env.act(action);
        }
    }

    fn observe(&self) -> [Step<Self::Observation>; N] {
        // TODO: Check the warning in Array::map about large array concern this function.
        let mut env = self.envs.iter();
        [(); N].map(|_| env.next().expect("we know the size of the env").observe())
    }
}

/// Blanket implementation for all Gym3Environment.
impl<E, const N: usize> ArrayEnv<E, N>
where
    E: Gym3Environment + Clone,
{
    //FIXME: if we want to initialize with different seed each env?
    // Create a new vec env.
    pub fn new(env: E) -> Self {
        Self {
            envs: [(); N].map(|_| env.clone()),
        }
    }
}

impl<E, const N: usize> FromIterator<E> for ArrayEnv<E, N>
where
    E: Gym3Environment,
{
    fn from_iter<T: IntoIterator<Item = E>>(iter: T) -> Self {
        let mut iter = iter.into_iter();
        Self {
            envs: [(); N].map(|_| iter.next().expect("we know the size of the env")),
        }
    }
}
