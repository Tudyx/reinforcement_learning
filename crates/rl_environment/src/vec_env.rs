use crate::{Gym3Environment, Step};

pub trait VecEnvironment {
    type Action;
    type Observation;

    fn act(&mut self, actions: Vec<Self::Action>);

    /// (Reward, observation first)
    /// The reward correspond to the previous action (not taken from the current observation)
    fn observe(&self) -> Vec<Step<Self::Observation>>;
}

pub struct VecEnv<E>
where
    E: Gym3Environment,
{
    envs: Vec<E>,
}

impl<E> VecEnvironment for VecEnv<E>
where
    E: Gym3Environment + Clone,
{
    type Action = E::Action;
    type Observation = E::Observation;

    fn act(&mut self, actions: Vec<Self::Action>) {
        assert_eq!(
            actions.len(),
            self.envs.len(),
            "You must provide the same number of actions that the number of envs."
        );

        for (action, env) in actions.into_iter().zip(self.envs.iter_mut()) {
            env.act(action);
        }
    }
    fn observe(&self) -> Vec<Step<Self::Observation>> {
        self.envs.iter().map(|env| env.observe()).collect()
    }
}

/// Blanket implementation for all Gym3Environment.
impl<E> VecEnv<E>
where
    E: Gym3Environment + Clone,
{
    //FIXME: if we want to initialize with different seed each env?
    // Create a new vec env.
    pub fn new(env: E, nb_env: usize) -> Self {
        Self {
            envs: vec![env; nb_env],
        }
    }
}
