//! Testing of the multi agent PettingZoo API.
//! https://pettingzoo.farama.org/content/environment_creation/

////////  Agent Environment Cycle (AEC) API ////////

enum Action {
    Rock,
    Paper,
    Scissors,
    None,
}

pub trait AecEnvironment {
    type Observation;
    type Action;

    fn observe(&self, agent: &str) -> Option<Self::Observation>;

    fn act(&mut self, action: Self::Action);
}

pub struct State(Action, Action);

struct RawEnv {
    state: State,
}

impl AecEnvironment for RawEnv {
    type Observation = (Action, Action);

    type Action = Action;

    fn observe(&self, agent: &str) -> Option<Self::Observation> {
        todo!()
    }

    fn act(&mut self, action: Self::Action) {
        todo!()
    }
}
