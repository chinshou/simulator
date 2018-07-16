import random
from v2 import run_benchmarks
from v2.ddqn_agent import DDQNAgent
random.seed(3456)

num_coins_per_order = 100 #0 means buy/sell all per order
recent_k = 500
epsilon_min = 0.0

coin_name = 'ethereum'

# run_benchmarks.run_bollingerband_agent(
#     coin_name=coin_name, num_coins_per_order = num_coins_per_order, recent_k=recent_k)
# run_benchmarks.run_random_agent(coin_name=coin_name, num_coins_per_order = num_coins_per_order, recent_k=recent_k)
# run_benchmarks.run_alwaysbuy_agent(coin_name=coin_name, num_coins_per_order = num_coins_per_order, recent_k=recent_k)

eth_agent = DDQNAgent(coin_name=coin_name, recent_k = 500, num_coins_per_order = num_coins_per_order,
                      epsilon_min = epsilon_min,
                      external_states = ["current_price", "cross_upper_band", "cross_lower_band"],
                      internal_states = ["is_holding_coin"], verbose = False)
# eth_agent.plot_env(states_to_plot=["current_price", "upper_band", "lower_band"])
# eth_agent.train(num_episodes=800)
# eth_agent.plot_cum_returns()
eth_agent.test(epsilon=0.018)