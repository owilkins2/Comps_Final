import agentpy as ap
import random
import matplotlib.pyplot as plt
import numpy as np


#parameters 
num_periods = 50
jump_start = 10
jump_end = 13
normal_increase = 1
jump_increase = 10
initial_fundamental_value = 100
proportion_extrapolative = 0.75
num_agents = 1000
knowledge_of_extrapolation = False



def get_extrapolative_component(price_history):
    expected_change = 0
    index = 1 # one represents current period
    while index <= 10:
        if len(price_history) > index + 1: # an entry does exist
            expected_change += price_history[len(price_history) - index]
        else: # an entry does not exist
            expected_change += (initial_fundamental_value - ((len(price_history) - index) * normal_increase))
        index += 1

    return expected_change



class Agent(ap.Agent):
    def setup(self):
        self.shares = random.randint(10, 100) 
        self.fundamental_value = initial_fundamental_value
        self.random_component = np.random.normal(loc=0, scale=2.0)
        self.belief = self.fundamental_value + self.random_component
        self.cash = 100000  # Cash available for trading
        self.fundamental = True if random.random() > proportion_extrapolative  else False

    def decide_trade(self, market_price):
        if self.belief > market_price and self.cash >= market_price:
            return "buy"
        elif self.belief < market_price and self.shares > 0:
            return "sell"
        else:
            return "hold"

    def execute_trade(self, trade, market_price):
        if trade == "buy":
            self.shares += 1
            self.cash -= market_price
        elif trade == "sell":
            self.shares -= 1
            self.cash += market_price




class StockMarketModel(ap.Model):
    def setup(self):
        self.agents = ap.AgentList(self, num_agents, Agent)
        self.market_price = initial_fundamental_value  # Initial market price
        self.price_history = []  # List to store the market price at each step
        self.fundamental_value_history = []

    def step(self):
        buy_val_diffs = []
        sell_val_diffs = []
        buy_count = 0
        sell_count = 0

        # Calculate buy/sell actions and value differences
        total_valuation = 0
        total_shares = 0
        for agent in self.agents:
            total_valuation += agent.belief
            total_shares += agent.shares
            trade = agent.decide_trade(self.market_price)
            if trade == "buy":
                buy_count += 1
                buy_val_diffs.append(agent.belief - self.market_price)
            elif trade == "sell":
                sell_count += 1
                sell_val_diffs.append(self.market_price - agent.belief)

        


        self.market_price = total_valuation / num_agents

        # Record market price
        self.price_history.append(self.market_price)
        self.fundamental_value_history.append(self.agents[0].fundamental_value)
        

        # Execute trades
        for agent in self.agents:
            agent.execute_trade(agent.decide_trade(self.market_price), self.market_price)

        # Increase fundamental value
        fundamental_sum = 0
        fundamental_count = 0
        extrapolative_sum = 0
        extrapolative_count = 0
        for agent in self.agents:
            #jump
            increase = 0
            if self.t in range(jump_start, jump_end):
                increase = jump_increase
            #normal
            else:
                increase = normal_increase

            agent.fundamental_value = agent.fundamental_value + increase 
            if agent.fundamental:
                if knowledge_of_extrapolation:
                    print("knowledge")
                    agent.belief = (1 - proportion_extrapolative) * (agent.fundamental_value + agent.random_component) + proportion_extrapolative * (agent.belief + (self.price_history[self.t - 1] - self.price_history[self.t - 2]))
                else:
                    print("no knowledge")
                    agent.belief = agent.fundamental_value + agent.random_component
                fundamental_sum += agent.belief
                fundamental_count += 1
            else:
                if self.t == 1:
                    agent.belief = agent.fundamental_value + agent.random_component
                else:
                    agent.belief = 0.2 * (agent.fundamental_value + agent.random_component) + 0.8 * (agent.belief + (self.price_history[self.t - 1] - self.price_history[self.t - 2]))
                extrapolative_sum += agent.belief
                extrapolative_count += 1
            
        #print ("/nfundamental value: " + str(self.agents[0].fundamental_value))
        #print ("price: " + str(self.price_history[self.t - 1]))
        #print ("avg fundamental belief: " + str(fundamental_sum/fundamental_count))
        #print ("avg extrapolative belief: " + str(extrapolative_sum/extrapolative_count))

    def end(self):
        bubble_size = calculate_bubble_size(self.price_history, self.fundamental_value_history)
        self.report("bubble_size", bubble_size)
        print(bubble_size)
        return bubble_size

def calculate_bubble_size(prices, fundamental_values):
    period = 0
    #print(prices)
    while not (period > jump_end and prices[period] > fundamental_values[period]):
        
        #print(str(period) + ": " + str(prices[period])) 
        period += 1
    bubble_size = prices[period] - fundamental_values[period]
    period += 1
    while prices[period] > fundamental_values[period] + 1:
        bubble_size += prices[period] - fundamental_values[period]
        period += 1
    return bubble_size
    


def get_percent_size_differences(parameters):
    proportions = []
    percentage_differences = []
    bubble_sizes_knowledge = []
    bubble_sizes_no_knowledge = []
    global proportion_extrapolative
    proportion_extrapolative = 0.25
    while proportion_extrapolative <= 1:
        print(f"Proportion Extrapolative: {proportion_extrapolative}%")

        # First run: No knowledge of extrapolation
        global knowledge_of_extrapolation
        knowledge_of_extrapolation = False
        market_model = StockMarketModel(parameters)  # New instance
        results = market_model.run()
        no_knowledge_bubble_size = results.reporters['bubble_size']
        bubble_sizes_no_knowledge.append(no_knowledge_bubble_size)

        # Second run: With knowledge of extrapolation
        knowledge_of_extrapolation = True
        market_model = StockMarketModel(parameters)  # New instance
        results = market_model.run()
        knowledge_bubble_size = results.reporters['bubble_size']
        bubble_sizes_knowledge.append(knowledge_bubble_size)

        # Calculate and store percentage difference
        percentage_difference = int(no_knowledge_bubble_size) / int(knowledge_bubble_size)
        percentage_differences.append(percentage_difference)
        proportions.append(proportion_extrapolative)

        # Increment proportion
        proportion_extrapolative = proportion_extrapolative + 0.01

    # Return results
    return proportions, percentage_differences, bubble_sizes_no_knowledge, bubble_sizes_knowledge


def graph_percentage_differences(proportions, percentage_differences):
    plt.figure(figsize=(10, 6))
    plt.scatter(proportions, percentage_differences)
    plt.title('Ratio of Bubble size w/o Knowlede of Extrapolation to Bubble Size w/ Knowledge by Proportion Extrapolative')
    plt.xlabel('Proportion Extrapolative')
    plt.ylabel('Bubble Size Ratio')
    plt.grid(True)
    plt.savefig("graph3")
    plt.show()

def graph_bubble_sizes(bubble_sizes_no_knowledge, bubble_sizes_knowledge):
    plt.figure(figsize=(10, 6))
    plt.scatter(proportions, bubble_sizes_no_knowledge, color="orange", label="No Knowledge of Extrapolation")
    plt.scatter(proportions, bubble_sizes_knowledge, color="green", label="Knowledge of Extrapolation")
    plt.title('Bubble Sizes w/ vs w/o knowledge of extrapolation by Proportion Extrapolative')
    plt.xlabel('Proportion Extrapolative')
    plt.ylabel('Bubble Size')
    plt.grid(True)
    plt.savefig("graph4")
    plt.show()


# Run the model
parameters = {"steps": 50}
market_model = StockMarketModel(parameters)
#results = market_model.run()
proportions, percentage_differences, bubble_sizes_no_knowledge, bubble_sizes_knowledge = get_percent_size_differences(parameters)
graph_percentage_differences(proportions, percentage_differences)
graph_bubble_sizes(bubble_sizes_no_knowledge, bubble_sizes_knowledge)


# Plot the market price over time
#plt.figure(figsize=(10, 6))
#plt.plot(market_model.price_history, color='blue', linewidth=2)
#plt.plot(market_model.fundamental_value_history, color='red', linewidth=2, linestyle='dashed')
#plt.title('Market Price Evolution')
#plt.xlabel('Step')
#plt.ylabel('Market Price')
#plt.grid(True)
#plt.show()



