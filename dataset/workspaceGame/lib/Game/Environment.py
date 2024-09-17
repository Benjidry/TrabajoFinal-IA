import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .Blackjack import BlackjackGame

class BJEnvironment(gym.Env):
    def __init__(self):
        self.game = BlackjackGame()
        self.deck_min_len = 109
        self.cart_counting = 0
        # player_sum, dealer_sum, usable_ace, has_double, prob_21, game_state
        # game_state = 0 normal, 1 double
        # 2 split. Not available in this version
        self.state_size = 6
        self.used_carts =[]
        # Hit, Stand, Double, Split
        self.action_size = 3
        self.observation_space = spaces.Box(low=np.array([4,2,0,0,0,0]), high=np.array([30,26,1,1,10,1]), shape=(self.state_size,), dtype=np.uint8)
        self.action_space = spaces.Discrete(self.action_size)

    @staticmethod
    def has_usable_ace(hand):
        """Check if the hand has a usable ace."""
        value, ace = 0, False
        for card in hand:
            card_number = card["number"]
            value += min(
                10, int(card_number) if card_number not in ["J", "Q", "K", "A"] else 11
            )
            ace |= card_number == "A"
        return int(ace and value + 10 <= 21)

    def get_badmove(self):
        return self.game.badMove

    def set_deck_per_game(self, deck):
        self.used_carts = np.copy(deck)

    def get_final_result(self):
        final_result = self.game.game_result()
        return final_result

    def step(self, action):
        act_string = ["hit", "stay", "double", "split"][action]
        state = self.get_obs()
        bet = self.game.bet_game
        status = self.game.player_action(act_string)

        reward = 0
        done = False

        bet = self.game.return_bounty(self.bet, act_string)

        if done or self.game.badMove:
            self.set_deck_per_game()

        if status[1] == "continue":
            if self.game.badMove:
                reward = -5
                done = True
            return state, action, reward, self.get_obs(), done

        # Si cuando el jugador o el dealer tienen 21 en la primera no obtiene recompensa
        if self.game.firstTurn:
            reward = 0
        else:
            if status[1] == "win":
                reward = bet / self.game.bet_game

            if status[1] == "loss":
                reward = -bet/self.game.bet_game

        if status[1] == "draw":
            reward = 0

        print(self.game.game_result())

        return state, action, reward, self.get_obs(), True

    def get_obs(self):
        """
        Devuelve la observación del estado del juego.
        Durante el turno del jugador, solo muestra la primera carta del dealer.
        Después del turno del jugador (cuando `lastTurn` es True), muestra todas las cartas del dealer.
        """
        player_sum = self.game.hand_value(self.game.player_hand)

        # Mostrar solo la primera carta del dealer mientras el jugador aún no ha terminado su turno
        if not self.game.lastTurn:
            dealer_card = self.game.hand_value(self.game.dealer_hand[:1])
        else:
            # Mostrar todas las cartas del dealer cuando el turno ha terminado
            dealer_card = self.game.hand_value(self.game.dealer_hand)

        usable_ace = self.has_usable_ace(self.game.player_hand)
        has_double = self.game.firstTurn
        game_state = self.game.status

        # Si el jugador ha hecho split y la primera mano se ha pasado, mostrar la segunda mano
        if game_state == 2 and player_sum > 21:
            player_sum = self.game.hand_value(self.game.splitted_hands[1])

        state = np.array(
            [
                player_sum,
                dealer_card,
                usable_ace,
                has_double,
                game_state,
            ]
        )
        state = state.astype(np.uint8)
        state = np.reshape(state, [1, self.state_size])
        return state

    def obtain_values(carta):
        value = 0

        if carta in ['2', '3', '4', '5', '6']:
            value = 1
        elif carta in ['7', '8', '9']:
            value = 0
        elif carta in ['10', 'J', 'Q', 'K', 'A']:
            value = -1
        else:
            value = 0
        return value

    def hilo_counting_start(carts):
    
        for cart in carts:
            count += obtain_values(cart['number'])

        return count


    def reset(self, bet):

        self.bet = bet

        if(len(self.game.get_deck()) == 312):
            self.set_deck_per_game(self.game.get_deck())

        if (len(self.game.get_deck()) <= self.deck_min_len):
            self.game.regenerate_deck()
            self.cart_counting = 0

        self.game.start_game(self.bet)
        hilo_counting_start(self.game.player_hand)
        hilo_counting_start(self.game.dealer_hand[0])

        self.status = ["act", "continue"]

        return self.get_obs(), {}
