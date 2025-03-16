import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cirq

def plot_results_cirq(results, labels, time_steps):
    plt.figure(figsize=(10, 6))
    
    for result, label in zip(results, labels):
        if isinstance(result, cirq.ResultDict):
            data = result.measurements['result']
        elif isinstance(result.data['result'], pd.Series):
            data = result.data['result'].values
        else:
            data = result.data['result']
        
        prob_0 = []
        for i in range(0, len(data), time_steps):
            chunk = data[i:i+time_steps]
            count_0 = np.sum(chunk == 0)
            if len(chunk) > 0:
                prob_0.append(count_0 / len(chunk))
            else:
                prob_0.append(0)  # Handle empty chunks to avoid division by zero
        
        plt.plot(range(len(prob_0)), prob_0, label=label)
    
    plt.xlabel('Measurement Cycle')
    plt.ylabel('Probability of |0>')
    plt.title('Qubit Fidelity During Measurement')
    plt.legend()
    plt.grid(True)
    plt.savefig('graphs/qubit_fidelity_plot.png')
    plt.show()

def print_result_info(result):
    print(f"Result type: {type(result)}")
    if isinstance(result, cirq.ResultDict):
        print(f"Params: {result.params}")
        print(f"Measurements keys: {result.measurements.keys()}")
        if 'result' in result.measurements:
            print(f"First few results: {result.measurements['result'][:10]}")
    elif hasattr(result, 'data'):
        print(f"Data keys: {result.data.keys()}")
        if 'result' in result.data:
            print(f"Result data type: {type(result.data['result'])}")
            print(f"First few results: {result.data['result'][:10]}")
    else:
        print("Unexpected result format")
