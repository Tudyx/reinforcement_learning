//! # Easy21 game
//!  This an implmentation if simplified version of blackjack taken from the David Silver course at the UCL univesity.
//! <https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=2>
//!
//!
//! The player objective is to have a total of card superior to the bank.

use crate::Environment;
use rand::prelude::*;
use std::{cmp::Ordering, ops::RangeInclusive};

// TODO: explore the "last_observe" pattern
// <https://github.com/christopher-hesse/computer-tennis/blob/9f8aeacf5240d616179fdadc4fc50c9fb15987b7/computer_tennis/env.py#L150>

/// Simplified version of BlackJack.
pub struct Easy21 {
    player_sum: i8,
    bank_sum: i8,
    first: bool,
    last_reward: f64, //TODO: explore this pattern
}

// Range where the sum of the cards is valid.
const CARDS_RANGE: RangeInclusive<i8> = 1..=21;

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum Action {
    /// Take a new card.
    Hit,
    /// Do nothing.
    Stick,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Observation {
    /// Sum of the player cards.
    pub player_sum: i8,
    /// Sum of the bank cards.
    pub bank_sum: i8,
}

impl Environment for Easy21 {
    type Action = Action;
    type Observation = Observation;

    fn act(&mut self, action: Self::Action) {
        if self.first {
            self.first = false;
        }

        match action {
            Action::Hit => {
                self.player_sum += self.draw_card();
                if self.is_player_busted() {
                    // println!("Player Busted! ({})", self.player_sum);
                    self.last_reward = -1.;
                    self.reset();
                } else {
                    self.last_reward = 0.;
                }
            }
            // End of the game.
            Action::Stick => {
                // The dealer play until to have at least a sum superior or equal to 17.
                while self.bank_sum < 17 && self.bank_sum > 0 {
                    self.bank_sum += self.draw_card();
                }

                if self.is_bank_busted() {
                    // println!("Bank Busted! ({})", self.bank_sum);
                    self.last_reward = 1.;
                } else {
                    match self.player_sum.cmp(&self.bank_sum) {
                        Ordering::Less => self.last_reward = -1.,
                        Ordering::Equal => self.last_reward = 0.,
                        Ordering::Greater => self.last_reward = 1.,
                    }
                }
                // We reset the env here. (act is the only mutable function)
                self.reset();
            }
        }
    }

    fn observe(&self) -> (f64, Self::Observation, bool) {
        let observation = Observation {
            player_sum: self.player_sum,
            bank_sum: self.bank_sum,
        };

        if !self.first {
            // println!("Player score {}", self.player_sum);
            // println!("Bank score {}", self.bank_sum);
        }

        (self.last_reward, observation, self.first)
    }

    fn metadata() -> crate::environment::Metadata {
        crate::environment::Metadata::default()
    }
}

impl Default for Easy21 {
    fn default() -> Self {
        Self {
            player_sum: thread_rng().gen_range(1..=10),
            bank_sum: thread_rng().gen_range(1..=10),
            first: true,
            last_reward: 0.0,
        }
    }
}

impl Easy21 {
    // Red has a probability of 1/3 and black 2/3.
    // Red are deduce from the total, black are added.
    fn draw_card(&self) -> i8 {
        let card = thread_rng().gen_range(1..=10);
        if thread_rng().gen::<f64>() < (2. / 3.) {
            card
        } else {
            -1 * card
        }
    }
    fn is_player_busted(&self) -> bool {
        !CARDS_RANGE.contains(&self.player_sum)
    }

    fn is_bank_busted(&self) -> bool {
        !CARDS_RANGE.contains(&self.bank_sum)
    }

    /// we reset everything except the winner, because we need it for the next observation
    /// to compute the reward.
    /// The player and th bank get one black card.
    fn reset(&mut self) {
        self.bank_sum = thread_rng().gen_range(1..=10);
        self.player_sum = thread_rng().gen_range(1..=10);
        self.first = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_validity_domain() {
        let mut env = Easy21::default();

        for _ in 0..1000 {
            assert!(!env.is_bank_busted());
            assert!(!env.is_player_busted());
            if rand::random() {
                env.act(Action::Hit);
            } else {
                env.act(Action::Stick);
            }
        }
    }

    #[test]
    fn color_distribution() {
        const CARD_NUMBER: u64 = 100_000;
        let mut num_red_cards = 0;
        let mut num_black_cards = 0;

        let env = Easy21::default();

        for _ in 0..CARD_NUMBER {
            let card = env.draw_card();

            if card.is_positive() {
                num_black_cards += 1;
            } else {
                num_red_cards += 1;
            }
        }

        let red_card_ratio = num_red_cards as f64 / CARD_NUMBER as f64;
        let black_card_ratio = num_black_cards as f64 / CARD_NUMBER as f64;

        println!("Percentage of red cards: {red_card_ratio}");
        println!("Percentage of black cards: {black_card_ratio}");

        assert!((0.30..=0.36).contains(&red_card_ratio));
        assert!((0.60..=0.70).contains(&black_card_ratio));
    }
}
