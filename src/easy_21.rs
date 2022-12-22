//! # Easy21 game
//!  This an implmentation if simplified version of blackjack taken from the David Silver course at the UCL univesity.
//! <https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=2>
//!
//!
//! The player objective is to have a total of card superior to the bank.

use crate::Environment;
use rand::prelude::*;
use std::cmp::Ordering;

pub struct Easy21 {
    player_cards: Cards,
    bank_cards: Cards,
    winner: Winner,
}

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
pub struct Card {
    number: u8,
    color: Color,
}

impl Card {
    fn new_random() -> Self {
        // FIXME: Red must have a probability of 1/3 and black 2/3.
        let color = if random() { Color::Red } else { Color::Black };

        Self {
            number: thread_rng().gen_range(1..=10),
            color,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Color {
    Red,
    Black,
}

pub enum Action {
    /// Take a new card.
    Hit,
    /// Do nothing.
    Stick,
}

#[derive(Clone)]
pub struct Observation {
    player_cards: Cards,
    bank_cards: Cards,
}

#[derive(Debug)]
pub enum Winner {
    Unknow,
    Player,
    Bank,
    Equality,
}

impl Environment for Easy21 {
    type Action = Action;
    type Observation = Observation;

    fn act(&mut self, action: Self::Action) {
        match action {
            Action::Hit => {
                self.player_cards.push(Card::new_random());
                if self.is_player_busted() {
                    self.winner = Winner::Bank;
                }
            }
            // End of the game.
            Action::Stick => {
                // The dealer play until to have at least a sum superior or equal to 17.
                while self.bank_cards.sum() < 17 {
                    self.bank_cards.push(Card::new_random());
                }

                if self.is_bank_busted() {
                    self.winner = Winner::Player;
                }

                match self.player_cards.sum().cmp(&self.bank_cards.sum()) {
                    Ordering::Less => self.winner = Winner::Bank,
                    Ordering::Equal => self.winner = Winner::Equality,
                    Ordering::Greater => self.winner = Winner::Player,
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

        match self.winner {
            Winner::Unknow => (0., observation, false),
            Winner::Player => (1., observation, true),
            Winner::Bank => (-1., observation, true),
            Winner::Equality => (0., observation, true),
        }
    }

    fn metadata() -> crate::environment::Metadata {
        todo!()
    }
}

impl Default for Easy21 {
    fn default() -> Self {
        Self {
            player_cards: Cards::default(),
            bank_cards: Cards::default(),
            winner: Winner::Unknow,
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

    fn reset(&mut self) {
        *self = Easy21::default();
    }
}

#[test]
fn rand_1_10() {
    println!("Dice roll: {}", thread_rng().gen_range(1..=10));
}
