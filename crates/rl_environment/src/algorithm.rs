// Represent an Rl algorithm

use serde::{Deserialize, Serialize};

// Est-ce qu'un algo peut-être générique sur les actions et les observation?

// Les algo definisse dan un tuple les upported action space. Grosse reflexion a avoir sur les action
// space.
// Dans python il semble avoir un "supported action space"

pub trait Algorithm {
    type Action;
    type Observation;

    type Model: Serialize;
    type Policy: Policy<Self::Observation, Self::Action>;

    fn save(&self) -> Result<(), ()>;
    fn load() -> Result<Self::Model, ()>;
    fn predict(&self, observation: Self::Observation) -> Self::Action {
        self.policy().predict(observation)
    }
    fn learn(&self, total_timesteps: usize, log_interval: Option<usize>) -> Self::Model;
    fn policy(&self) -> &Self::Policy;
}

// Une policy par algo mais une policy peut-être partagé par plusieurs algo

pub trait Policy<O, A> {
    fn predict(&self, observation: O) -> A;
}

trait OnPolicyAlgorithm: Algorithm {
    fn train(&self);
    fn num_timesteps(&self) -> usize;
}

// impl<T> Algorithm for T
// where
//     T: OnPolicyAlgorithm,
// {
//     type Action = i32;
//     type Observation = usize;

//     type Model = ();

//     type Policy = ();

//     fn save(&self) -> Result<(), ()> {
//         todo!()
//     }

//     fn load() -> Result<Self::Model, ()> {
//         todo!()
//     }

//     fn learn(&self, total_timesteps: usize, log_interval: Option<usize>) -> Self::Model {
//         let mut iteration = 0;
//         while self.num_timesteps() < total_timesteps {
//             iteration += 1;
//             if let Some(log_interval) = log_interval {
//                 if log_interval % iteration == 0 {
//                     // log time elpased
//                     // log fps
//                     // log time elapsed
//                 }
//             }
//             self.train();
//         }

//         todo!()
//     }

//     fn policy(&self) -> &Self::Policy {
//         todo!()
//     }
// }
