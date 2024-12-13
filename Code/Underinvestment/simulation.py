import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
import time

class ConsumerAgent(ap.Agent):
    
    def setup(self):
        self.wealth = 1000
        self.salary = 50000
        self.interest_rate = 0.08
        self.age = 22
        self.consumption = 0
        self.social_security = 22000



    def calculate_utility(self, current_consumption, change_factor, age):
      
        remaining_wealth = self.wealth
        current_age = age
        total_utility = 0
        discount_rate = 0.95 
        beta = 1   # Present bias parameter


        while current_age <= 85 and current_age < age + 10:
            # Add salary during working years
            if current_age < 65:
                remaining_wealth += self.salary
            else:
                remaining_wealth += self.social_security

            # Subtract consumption from wealth
            if current_age == age:
                remaining_wealth -= current_consumption
            else:
                remaining_wealth -= (current_consumption + (change_factor * (age - current_age)))

            # If wealth goes negative, i.e. infeasible
            if remaining_wealth < 0:
                return -99999999

            # Apply interest rate
            remaining_wealth *= (1.5 + self.interest_rate)

            # Calculate utility for this period and add to total utility
            risk_aversion = 2
            local_discount_rate = (discount_rate ** (current_age - age)) * beta
            if current_age != age:
                #total_utility += (future_consumption ** (1 - risk_aversion)) / (1 - risk_aversion) * discount_rate
                total_utility += np.sqrt(current_consumption + (change_factor * (age - current_age))) * local_discount_rate
            else:
                #total_utility -= (current_consumption ** (1 - risk_aversion)) / (1 - risk_aversion) * local_discount_rate
                total_utility += np.sqrt(current_consumption)
            

            # Increment age
            current_age += 1
            

        return total_utility

    def make_consumption_decision(self):
        current_options = None
        if self.age < 65:
            current_options = np.linspace(35000, self.wealth + self.salary, 50)
        else:
            current_options = np.linspace(35000, self.wealth, 50)
        
        current_options = np.linspace(0, 200000, 100)
        future_options = np.linspace(0, 200000, 100)
        change_factors = np.linspace(-1000, 1000, 100)

        max_utility = -np.inf
        consumption_choice = 0

        for current_option in current_options:
            
            for change_factor in change_factors:
                utility = self.calculate_utility(current_option, change_factor, self.age)
                if utility == -99999999:
                    break
                #print('age:' + str(self.age) + ", current option: " + str(current_option) + ", future option: " + str(future_option) + ", utility: " + str(utility))
                if utility > max_utility:
                    chosen_future = change_factor
                    max_utility = utility
                    consumption_choice = current_option

                #print(str(self.age) + ": option " + str(option) + " = " +str(utility)) 

        print(str(consumption_choice))
        time.sleep(5)
        return consumption_choice
        

    def update(self):
        self.consumption = self.make_consumption_decision()
        self.wealth -= self.consumption
        
        if self.age < 65: # Working years
            self.wealth += self.salary
        else:
            self.wealth += self.social_security

        self.wealth *= (1 + self.interest_rate)
        self.age += 1

class ConsumerModel(ap.Model):

    def setup(self):
        self.agents = ap.AgentList(self, 1, ConsumerAgent)
        self.wealth = []
        self.consumption = []

    def step(self):
        self.agents.update()

        self.wealth.append(self.agents[0].wealth)
        self.consumption.append(self.agents[0].consumption)

    def end(self):
        self.record('wealth', self.wealth)
        self.record('consumption', self.consumption)

# Run Simulation
parameters = {'steps': 64}  # Simulate from age 22 to 85
model = ConsumerModel(parameters)
results = model.run()

# Data for Plot
wealth = results.variables['ConsumerModel']['wealth']
consumption = results.variables['ConsumerModel']['consumption']
wealth = [item for sublist in wealth for item in sublist]
consumption = [item for sublist in consumption for item in sublist]
years = np.arange(22, 22 + len(wealth))

print("consumption: " + str(consumption))
print("wealth: " + str(wealth))

# Plot Results
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(years, wealth, marker='o', color='tab:blue', label='Wealth')
ax1.set_xlabel('Age', fontsize=14)
ax1.set_ylabel('Wealth', color='tab:blue', fontsize=14)
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.set_ylim([0,100000])
ax2.plot(years, consumption, marker='x', color='tab:orange', label='Consumption')
ax2.set_ylabel('Consumption', color='tab:orange', fontsize=14)
ax2.tick_params(axis='y', labelcolor='tab:orange')
plt.title('Wealth and Consumption by Age', fontsize=16)
ax1.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.savefig('beta=0.7 delta=0.95.png')
plt.show()
