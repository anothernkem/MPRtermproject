import math
import pandas as pd
import numpy as np
from tabulate import tabulate
import os


class DynamicMRP:
    """
    Enhanced MRP calculator with forward-looking net requirements calculation
    """

    def __init__(self, lead_time, safety_stock, holding_cost, setup_cost, periods=10):
        # Initial configuration
        self.initial_lead_time = lead_time
        self.initial_safety_stock = safety_stock
        self.initial_holding_cost = holding_cost
        self.initial_setup_cost = setup_cost

        # Configurable parameters
        self.periods = periods
        self.lead_time = lead_time
        self.safety_stock = safety_stock
        self.holding_cost = holding_cost
        self.setup_cost = setup_cost

        # Tracking variables
        self.demand = [0] * periods
        self.starting_inventory = 0
        self.current_period = 0

    def load_from_csv(self, file_path):
        """
        Load MRP input parameters from CSV file
        """
        try:
            data = pd.read_csv(file_path)

            required_columns = ['Demand', 'Lead Time', 'Safety Stock', 'Starting Inventory', 'Holding Cost',
                                'Setup Cost']
            for col in required_columns:
                if col not in data.columns:
                    print(f"Error: Missing required column '{col}' in CSV file.")
                    return False

            demand = data['Demand'].dropna().tolist()
            lead_time = int(data['Lead Time'].dropna().iloc[0])
            safety_stock = int(data['Safety Stock'].dropna().iloc[0])
            starting_inventory = int(data['Starting Inventory'].dropna().iloc[0])
            holding_cost = float(data['Holding Cost'].dropna().iloc[0])
            setup_cost = float(data['Setup Cost'].dropna().iloc[0])

            self.periods = len(demand)
            self.lead_time = lead_time
            self.safety_stock = safety_stock
            self.holding_cost = holding_cost
            self.setup_cost = setup_cost
            self.demand = demand
            self.starting_inventory = starting_inventory

            print("CSV file loaded successfully!")
            return True

        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return False
        except Exception as e:
            print(f"An error occurred while loading the CSV: {e}")
            return False

    def set_initial_conditions(self, demand, starting_inventory):
        """Set initial demand and starting inventory"""
        self.demand = demand.copy()
        self.starting_inventory = starting_inventory

    def update_demand(self, period, new_demand):
        """Update demand for a specific period"""
        if 0 <= period < self.periods:
            self.demand[period] = new_demand
            return True
        return False

    def calculate_mrp(self, technique, fixed_quantity=None):
        """Calculate MRP using specified technique with forward-looking net requirements"""
        projected_inventory = [self.starting_inventory] + [0] * (self.periods - 1)
        planned_orders = [0] * self.periods
        scheduled_receipts = [0] * self.periods
        net_requirements = [0] * self.periods
        lot_sizes = [0] * self.periods
        holding_costs = [0] * self.periods
        setup_costs = [0] * self.periods

        average_demand = sum(self.demand) / len(self.demand)
        eoq = self._calculate_eoq(average_demand) if technique == "EOQ" else None

        current_inventory = self.starting_inventory
        total_holding_cost = 0
        total_setup_cost = 0

        for period in range(self.periods):
            if period > 0:
                projected_inventory[period] = projected_inventory[period - 1] + scheduled_receipts[period - 1] - \
                                              self.demand[period - 1]

            # Calculate total future demand considering lead time
            future_demand = 0
            future_periods = min(period + self.lead_time + 1, self.periods)
            for future_period in range(period, future_periods):
                future_demand += self.demand[future_period]

            # Calculate net requirements considering future demand and safety stock
            if projected_inventory[period] <= self.safety_stock:
                net_requirements[period] = future_demand + (self.safety_stock - projected_inventory[period])
            else:
                if (projected_inventory[period] - future_demand) < self.safety_stock:
                    net_requirements[period] = future_demand - (projected_inventory[period] - self.safety_stock)
                else:
                    net_requirements[period] = 0

            lot_size = self._determine_lot_size(
                period,
                projected_inventory,
                net_requirements,
                technique,
                eoq,
                fixed_quantity
            )

            if lot_size > 0:
                lot_sizes[period] = lot_size
                future_period = min(period + self.lead_time, self.periods - 1)
                scheduled_receipts[future_period] += lot_size
                setup_costs[period] = self.setup_cost
                planned_orders[period] = lot_size

            current_inventory = projected_inventory[period] + scheduled_receipts[period] - self.demand[period]
            holding_cost = max(0, current_inventory) * self.holding_cost
            holding_costs[period] = holding_cost

            total_holding_cost += holding_cost
            total_setup_cost += setup_costs[period]

            if period < self.periods - 1:
                projected_inventory[period + 1] = current_inventory

        return {
            'technique': technique,
            'projected_inventory': projected_inventory,
            'planned_orders': planned_orders,
            'scheduled_receipts': scheduled_receipts,
            'net_requirements': net_requirements,
            'lot_sizes': lot_sizes,
            'holding_costs': holding_costs,
            'setup_costs': setup_costs,
            'total_holding_cost': total_holding_cost,
            'total_setup_cost': total_setup_cost,
            'total_cost': total_holding_cost + total_setup_cost,
            'safety_stock_maintained': all(inv >= self.safety_stock for inv in projected_inventory)
        }

    def _determine_lot_size(self, period, projected_inventory, net_requirements, technique, eoq, fixed_quantity):
        """Determine lot size based on chosen technique"""
        if net_requirements[period] <= 0:
            return 0

        if technique == "Lot-for-Lot":
            return net_requirements[period]

        elif technique == "EOQ":
            return max(eoq, net_requirements[period])

        elif technique == "Fixed Order Quantity":
            return max(fixed_quantity, net_requirements[period])

        return 0

    def _calculate_eoq(self, average_demand):
        """Calculate Economic Order Quantity"""
        return math.ceil(math.sqrt((2 * average_demand * self.setup_cost) / self.holding_cost))

    def display_mrp_results(self, mrp_results):
        """Display MRP calculation results with periods shown horizontally"""
        print(f"\n--- MRP Results for {mrp_results['technique']} Technique ---")
        print(f"Safety Stock Level: {self.safety_stock}")
        print(f"Lead Time: {self.lead_time}")

        # Create period headers
        period_headers = ["Period"] + [str(i + 1) for i in range(self.periods)]

        # Create rows for each metric
        rows = [
            ["Demand"] + self.demand,
            ["Net Requirements"] + mrp_results['net_requirements'],
            ["Lot Size"] + mrp_results['lot_sizes'],
            ["Scheduled Receipts"] + mrp_results['scheduled_receipts'],
            ["Projected Inventory"] + [
                f"{inv}*" if inv < self.safety_stock else str(inv)
                for inv in mrp_results['projected_inventory']
            ],
            ["Holding Cost"] + mrp_results['holding_costs'],
            ["Setup Cost"] + mrp_results['setup_costs']
        ]

        print(tabulate(rows, headers=period_headers, tablefmt="grid"))
        print(f"\nTotal Holding Cost: {mrp_results['total_holding_cost']:.2f}")
        print(f"Total Setup Cost: {mrp_results['total_setup_cost']:.2f}")
        print(f"Total Cost: {mrp_results['total_cost']:.2f}")

        if not mrp_results['safety_stock_maintained']:
            print("\nWarning: Safety stock level was not maintained in all periods!")


def interactive_mrp_planner():
    """Interactive MRP planning interface"""
    print("Dynamic MRP Calculator")

    mrp = None

    while True:
        print("\n--- Main Menu ---")
        print("1. Load Data from CSV")
        print("2. Manual Data Entry")
        print("3. Exit")

        choice = input("Enter your choice (1-3): ")

        if choice == '1':
            while True:
                file_path = input("Enter the full path to your CSV file: ")
                file_path = os.path.normpath(file_path)

                mrp = DynamicMRP(lead_time=1, safety_stock=0, holding_cost=1, setup_cost=50)

                if mrp.load_from_csv(file_path):
                    break
                else:
                    retry = input("Would you like to try again? (y/n): ").lower()
                    if retry != 'y':
                        break

        elif choice == '2':
            periods = int(input("Enter number of planning periods: "))
            lead_time = int(input("Enter lead time: "))
            safety_stock = int(input("Enter safety stock level: "))
            holding_cost = float(input("Enter holding cost per unit: "))
            setup_cost = float(input("Enter setup cost: "))
            starting_inventory = int(input("Enter starting inventory: "))

            mrp = DynamicMRP(lead_time, safety_stock, holding_cost, setup_cost, periods)

            initial_demand = []
            for i in range(periods):
                demand = int(input(f"Enter demand for period {i + 1}: "))
                initial_demand.append(demand)

            mrp.set_initial_conditions(initial_demand, starting_inventory)

        elif choice == '3':
            print("Exiting MRP Planner. Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")
            continue

        while mrp is not None:
            print("\n--- MRP Operations ---")
            print("1. View Current Parameters")
            print("2. Edit Demand")
            print("3. Edit Safety Stock")
            print("4. Calculate MRP")
            print("5. Compare Techniques")
            print("6. Return to Main Menu")

            sub_choice = input("Enter your choice (1-6): ")

            if sub_choice == '1':
                print("\nCurrent Parameters:")
                print(f"Safety Stock Level: {mrp.safety_stock}")
                print(f"Lead Time: {mrp.lead_time}")
                print(f"Holding Cost: ${mrp.holding_cost}")
                print(f"Setup Cost: ${mrp.setup_cost}")
                print("\nCurrent Demand:")
                for i, demand in enumerate(mrp.demand, 1):
                    print(f"Period {i}: {demand}")

            elif sub_choice == '2':
                period = int(input(f"Enter period to edit (1-{mrp.periods}): ")) - 1
                new_demand = int(input("Enter new demand: "))
                if mrp.update_demand(period, new_demand):
                    print("Demand updated successfully!")
                else:
                    print("Invalid period!")

            elif sub_choice == '3':
                current_ss = mrp.safety_stock
                print(f"\nCurrent Safety Stock Level: {current_ss}")
                new_ss = int(input("Enter new safety stock level: "))
                if new_ss >= 0:
                    mrp.safety_stock = new_ss
                    print("Safety stock updated successfully!")
                else:
                    print("Invalid safety stock level! Must be non-negative.")

            elif sub_choice == '4':
                technique = input("Choose technique (Lot-for-Lot/EOQ/Fixed Order Quantity): ")
                if technique in ["Lot-for-Lot", "EOQ", "Fixed Order Quantity"]:
                    fixed_quantity = None
                    if technique == "Fixed Order Quantity":
                        fixed_quantity = int(input("Enter fixed order quantity: "))
                    results = mrp.calculate_mrp(technique, fixed_quantity)
                    mrp.display_mrp_results(results)
                else:
                    print("Invalid technique!")

            elif sub_choice == '5':
                fixed_quantity = int(input("Enter fixed order quantity for FOQ comparison: "))
                print("Comparing all techniques:")
                technique_results = []
                for tech in ["Lot-for-Lot", "EOQ", "Fixed Order Quantity"]:
                    results = mrp.calculate_mrp(tech, fixed_quantity if tech == "Fixed Order Quantity" else None)
                    technique_results.append(results)
                    mrp.display_mrp_results(results)

                best_technique = min(technique_results, key=lambda x: x['total_cost'])
                print(f"\nMost Cost-Effective Technique: {best_technique['technique']}")

            elif sub_choice == '6':
                break

            else:
                print("Invalid choice. Please try again.")


if __name__ == "__main__":
    interactive_mrp_planner()