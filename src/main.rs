// In python you can portailly update that's why environment initialization is divided into the constructor and the reset function.
// They choose to use an array of 2 elements to represent the cordonate in an array of 2 dimension. That's probably because they want to use tensor later.

// TODO: manage la seed.

// open question
// - Est-ce que je peux utiliser de trucs plus idiomatque pour les enum? Je pourrait faire des "From"
// - Est ce que l'action space je peux le definir à partir d'un associated constant?
// - Est ce que je fois parallelizer l'environment pour voir si cela raoute des contarintes?
mod algorithm;
mod easy_21;
mod environment;
pub use environment::Environment;

use anyhow::bail;
mod grid_word;
mod viewer;
mod wrapper;

use easy_21::Action;
use easy_21::Easy21;

use std::io;

struct Interactive {
    env: Easy21,
    total_steps: usize,
    episode_steps: usize,
}

impl Interactive {
    fn new(env: Easy21) -> Self {
        Self {
            env,
            total_steps: 0,
            episode_steps: 0,
        }
    }

    fn run(&mut self) -> anyhow::Result<()> {
        let (_, observation, _) = self.env.observe();
        dbg!(observation);
        loop {
            self.update()?;
        }
    }

    fn retrieve_key_pressed(&self) -> anyhow::Result<<Easy21 as Environment>::Action> {
        let mut action = String::new();
        io::stdin().read_line(&mut action)?;
        let action = action.trim().parse::<usize>()?;
        match action {
            0 => Ok(Action::Hit),
            1 => Ok(Action::Stick),
            _ => bail!("Wrong key!"),
        }
    }

    fn update(&mut self) -> anyhow::Result<()> {
        let action = self.retrieve_key_pressed()?;
        let first = self.act(action);

        if first {
            self.episode_steps = 0;
        }

        Ok(())
    }
    fn act(&mut self, action: <Easy21 as Environment>::Action) -> bool {
        self.env.act(action);

        self.episode_steps += 1;
        self.total_steps += 1;
        let (reward, observation, first) = self.env.observe();
        println!("Reward: {reward}");
        if first {
            println!("Finish the episode in {} steps", self.episode_steps);
        }
        dbg!(observation);
        first
    }
}

fn main() -> anyhow::Result<()> {
    let mut agent = Interactive::new(Easy21::default());
    agent.run()?;
    Ok(())
}
