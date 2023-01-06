use std::io;

use anyhow::bail;

use rust_gym::easy_21::{Action, Easy21};
use rust_gym::Gym3Environment;

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

    fn retrieve_key_pressed(&self) -> anyhow::Result<<Easy21 as Gym3Environment>::Action> {
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
        println!("Enter action:");
        let action = self.retrieve_key_pressed()?;
        let first = self.act(action);

        if first {
            self.episode_steps = 0;
        }

        Ok(())
    }
    fn act(&mut self, action: <Easy21 as Gym3Environment>::Action) -> bool {
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
