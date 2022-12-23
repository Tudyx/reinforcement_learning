//! # Easy21 game
//!  This an implmentation if simplified version of blackjack taken from the David Silver course at the UCL univesity.
//! <https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=2>
//!
//!
//! The player objective is to have a total of card superior to the bank.

use crate::Environment;
use rand::prelude::*;
use std::cmp::Ordering;

// TODO: explore the "last_observe" pattern
// <https://github.com/christopher-hesse/computer-tennis/blob/9f8aeacf5240d616179fdadc4fc50c9fb15987b7/computer_tennis/env.py#L150>

/// Simplified version of BlackJack.
pub struct Easy21 {
    player_cards: Cards,
    bank_cards: Cards,
    winner: Winner,
    first: bool,
    last_reward: f64, //TODO: explore this pattern
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
struct Cards(Vec<Card>);

impl Cards {
    fn sum(&self) -> i32 {
        self.0
            .iter()
            .map(|card| match card.color {
                Color::Black => card.number as i32,
                Color::Red => -1 * card.number as i32,
            })
            .sum()
    }

    fn is_busted(&self) -> bool {
        !(1..=21).contains(&self.sum())
    }

    fn push(&mut self, card: Card) {
        self.0.push(card);
    }
}

impl Default for Cards {
    /// At the start of the game both the player and the dealer draw one black
    /// card (fully observed)
    fn default() -> Self {
        Self(vec![Card {
            color: Color::Black,
            number: thread_rng().gen_range(1..=10),
        }])
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Card {
    number: u8,
    color: Color,
}

impl Card {
    fn new_random() -> Self {
        //  Red has a probability of 1/3 and black 2/3.
        let color = match thread_rng().gen_range(1..=3) {
            1 => Color::Red,
            2..=3 => Color::Black,
            _ => unreachable!(),
        };

        Self {
            number: thread_rng().gen_range(1..=10),
            color,
        }
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum Color {
    Red,
    Black,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum Action {
    /// Take a new card.
    Hit,
    /// Do nothing.
    Stick,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Observation {
    player_cards: Cards,
    bank_cards: Cards,
}

#[derive(Debug)]
pub enum Winner {
    Unknown,
    Player,
    Bank,
    Equality,
}

impl Environment for Easy21 {
    type Action = Action;
    type Observation = Observation;

    fn act(&mut self, action: Self::Action) {
        if self.first {
            self.winner = Winner::Unknown;
            self.first = false;
        }

        match action {
            Action::Hit => {
                self.player_cards.push(Card::new_random());
                if self.is_player_busted() {
                    println!("Player Busted! ({})", self.player_cards.sum());
                    self.winner = Winner::Bank;
                    self.reset();
                }
            }
            // End of the game.
            Action::Stick => {
                // The dealer play until to have at least a sum superior or equal to 17.
                while self.bank_cards.sum() < 17 {
                    self.bank_cards.push(Card::new_random());
                }

                if self.is_bank_busted() {
                    println!("Bank Busted! ({})", self.bank_cards.sum());
                    self.winner = Winner::Player;
                } else {
                    match self.player_cards.sum().cmp(&self.bank_cards.sum()) {
                        Ordering::Less => self.winner = Winner::Bank,
                        Ordering::Equal => self.winner = Winner::Equality,
                        Ordering::Greater => self.winner = Winner::Player,
                    }
                }
                // We reset the env here. (act is the only mutable function)
                self.reset();
            }
        }
    }

    fn observe(&self) -> (f64, Self::Observation, bool) {
        let observation = Observation {
            player_cards: self.player_cards.clone(),
            bank_cards: self.bank_cards.clone(),
        };

        if !self.first {
            println!("Player score {}", self.player_cards.sum());
            println!("Bank score {}", self.bank_cards.sum());
        }

        let reward = match self.winner {
            Winner::Unknown => 0.,
            Winner::Equality => 0.,
            Winner::Player => 1.,
            Winner::Bank => -1.,
        };

        (reward, observation, self.first)
    }

    fn metadata() -> crate::environment::Metadata {
        crate::environment::Metadata::default()
    }
}

impl Default for Easy21 {
    fn default() -> Self {
        Self {
            player_cards: Cards::default(),
            bank_cards: Cards::default(),
            winner: Winner::Unknown,
            first: true,
            last_reward: 0.0,
        }
    }
}

impl Easy21 {
    fn is_player_busted(&self) -> bool {
        self.player_cards.is_busted()
    }

    fn is_bank_busted(&self) -> bool {
        self.bank_cards.is_busted()
    }

    /// we reset everything except the winner, because we need it for the next observation
    /// to compute the reward.
    fn reset(&mut self) {
        self.player_cards = Cards::default();
        self.bank_cards = Cards::default();
        self.first = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum() {
        let cards = Cards(vec![
            Card {
                color: Color::Black,
                number: 10,
            },
            Card {
                color: Color::Red,
                number: 2,
            },
        ]);

        assert_eq!(cards.sum(), 8);
    }

    #[test]
    fn color_distribution() {
        const CARD_NUMBER: u64 = 100_000;
        let mut num_red_cards = 0;
        let mut num_black_cards = 0;

        for _ in 0..CARD_NUMBER {
            let card = Card::new_random();
            match card.color {
                Color::Red => num_red_cards += 1,
                Color::Black => num_black_cards += 1,
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
