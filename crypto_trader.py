import random
from v2 import run_benchmarks
random.seed(3456)

num_coins_per_order = 100 #0 means buy/sell all per order
recent_k = 500
epsilon_min = 0.0

coin_name = 'ethereum'

run_benchmarks.run_bollingerband_agent(
    coin_name=coin_name, num_coins_per_order = num_coins_per_order, recent_k=recent_k)
run_benchmarks.run_random_agent(coin_name=coin_name, num_coins_per_order = num_coins_per_order, recent_k=recent_k)
run_benchmarks.run_alwaysbuy_agent(coin_name=coin_name, num_coins_per_order = num_coins_per_order, recent_k=recent_k)
