import numpy as np
import matplotlib.pyplot as plt

def utility_function(consumption, risk_aversion=2):
    if consumption <= 0:
        return -np.inf  # Disallow invalid consumption
    return (consumption ** (1 - risk_aversion)) / (1 - risk_aversion)

def hyperbolic_discount(beta, delta, t):
    return beta * (delta ** t) if t > 0 else 1

def life_cycle_simulation(
    initial_wealth, salary, retirement_age, end_age, 
    beta, delta, investment_return, gamma=2
):

    num_periods = end_age - 20 + 1
    wealth_plan = [initial_wealth]
    consumption_plan = []

    # Backward induction to compute optimal policies
    value_functions = [{} for _ in range(num_periods)]
    for t in range(num_periods - 1, -1, -1):
        print(t)
        is_working = (20 + t) < retirement_age
        for current_wealth in range(1, int(initial_wealth * 10) + 1):
            max_utility = float("-inf")
            optimal_consumption = 0
            consumption_choices = np.linspace(1, current_wealth / 10, min(50, current_wealth))
            for consumption in consumption_choices:
                remaining_wealth = max((current_wealth / 10 - consumption) * (1 + investment_return), 0)
                next_wealth = int(remaining_wealth * 10)
                future_utility = value_functions[t + 1].get(next_wealth, 0) if t + 1 < num_periods else 0
                total_utility = utility_function(consumption, gamma) + hyperbolic_discount(beta, delta, 1) * future_utility
                if total_utility > max_utility:
                    max_utility = total_utility
                    optimal_consumption = consumption
            value_functions[t][current_wealth] = optimal_consumption

    # Forward simulation to determine wealth and consumption levels
    current_wealth = initial_wealth
    for t in range(num_periods):
        optimal_consumption = value_functions[t].get(int(current_wealth * 10), 0)
        consumption_plan.append(optimal_consumption)
        wealth_plan.append(current_wealth)
        current_wealth = max((current_wealth - optimal_consumption) * (1 + investment_return), 0)
        if (20 + t) < retirement_age:
            current_wealth += salary

    return consumption_plan, wealth_plan

# Parameters
initial_wealth = 10000
salary = 50000
start_age = 20
retirement_age = 65
end_age = 85
investment_return = 0.05
delta = 0.97
beta_with_bias = 0.7
beta_no_bias = 1.0

# Simulations
consumption_with_bias, wealth_with_bias = life_cycle_simulation(
    initial_wealth, salary, retirement_age, end_age, 
    beta_with_bias, delta, investment_return
)

consumption_no_bias, wealth_no_bias = life_cycle_simulation(
    initial_wealth, salary, retirement_age, end_age, 
    beta_no_bias, delta, investment_return
)

# Plotting
ages = np.arange(start_age, end_age + 1)
plt.figure(figsize=(12, 6))

# Plot with bias
plt.plot(ages, consumption_with_bias, label="Consumption (With Bias)", linestyle="--")
plt.plot(ages, wealth_with_bias, label="Wealth (With Bias)", linestyle="--")

# Plot without bias
plt.plot(ages, consumption_no_bias, label="Consumption (No Bias)", linestyle="-")
plt.plot(ages, wealth_no_bias, label="Wealth (No Bias)", linestyle="-")

plt.title("Consumption and Wealth Over Life Cycle")
plt.xlabel("Age")
plt.ylabel("Amount ($)")
plt.legend()
plt.grid()
plt.show()
