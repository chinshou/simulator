import random
from v2 import run_benchmarks
from v2.ddqn_agent import DDQNAgent
import profile

num_coins_per_order = 10 #0 means buy/sell all per order
epsilon_min = 0.0

coin_name = 'bitcoin'

# run_benchmarks.run_bollingerband_agent(
#     coin_name=coin_name, num_coins_per_order = num_coins_per_order, recent_k=recent_k)
# run_benchmarks.run_random_agent(coin_name=coin_name, num_coins_per_order = num_coins_per_order, recent_k=recent_k)
# run_benchmarks.run_alwaysbuy_agent(coin_name=coin_name, num_coins_per_order = num_coins_per_order, recent_k=recent_k)

btc_agent = DDQNAgent(coin_name=coin_name, num_step= 8640, num_coins_per_order = num_coins_per_order,
                      epsilon_min = epsilon_min,
                      external_states = ["current_price", "dif5t","dea5t", "volume"],
                      internal_states = ["is_holding_coin"], verbose = False)
# btc_agent.plot_env(states_to_plot=["current_price", "upper_band", "lower_band"])
#btc_agent.train(num_episodes=60)
#profile.run("btc_agent.train(num_episodes=80)")
#btc_agent.plot_cum_returns()
btc_agent.test(epsilon=0.018)
#btc_agent.test(epsilon=0.01)