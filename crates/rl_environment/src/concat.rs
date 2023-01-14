use crate::Environment;

struct ConcatEnv<E> {
    envs: Vec<E>,
}

impl<E> ConcatEnv<E> {
    fn new(envs: Vec<E>) -> Self {
        Self { envs }
    }
}

impl<E> Environment for ConcatEnv<E>
where
    E: Environment,
{
    type Action = Vec<E::Action>;

    type Observation = Vec<E::Observation>;

    fn act(&mut self, action: Self::Action) {
        for (env, action) in self.envs.iter_mut().zip(action.into_iter()) {
            env.act(action);
        }
    }

    // marche pas avec l'abstraction actuelle:

    // Est-ce qu'il faut faire une autre abstraction genre VecEnv ?
    // avoir un associated type Step pourrait être une bonne idée également.

    fn observe(&self) -> (crate::environment::Reward, Self::Observation, bool) {
        let mut observations = Vec::new();
        for env in self.envs {
            observations.push(env.observe())
        }
        observations
    }

    fn metadata() -> crate::environment::Metadata {
        todo!()
    }
}
