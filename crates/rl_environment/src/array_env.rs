use crate::{environment::Reward, Gym3Environment};

pub trait ArrayEnvironment<const N: usize> {
    type Action;
    type Observation;

    fn act(&mut self, actions: [Self::Action; N]);

    /// The reward correspond to the previous action (not taken from the current observation)
    fn observe(&self) -> [(Reward, Self::Observation, bool); N];
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

    fn observe(&self) -> [(Reward, Self::Observation, bool); N] {
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
