import json
import operator
import os

from matplotlib import pyplot as plt
import numpy as np


class Metrics:
    def __init__(self):
        self.store = {'default': {}}
        self.keys = []
        self.namespaces = ['default']

    def register_namespace(self, namespace):
        self.namespaces.append(namespace)
        self.store[namespace] = {}

    def _register_key(self, key, namespace):
        self.keys.append(key)
        self.store[namespace][key] = []

    def add(self, key, value, namespace=None):
        if namespace is None:
            namespace = self.namespaces[-1]

        if key not in self.keys:
            self._register_key(key, namespace)

        self.store[namespace][key].append(value)

    def save(self, path: str):
        with open(os.path.join(path, 'metrics.json'), 'w') as f:
            json.dump(self.store, f, indent=4)

    def summary(self, path: str, plots=[], additional_stats=[], plot_compress: int = None):
        summary = {}

        ops = {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv
        }
        
        for namespace in self.namespaces:
            stats = {}
            # variants: avg, max, min, std, last
            for key in self.keys:
                stats[key + '_avg'] = np.mean(self.store[namespace][key])
                stats[key + '_max'] = float(np.max(self.store[namespace][key]))
                stats[key + '_min'] = float(np.min(self.store[namespace][key]))
                stats[key + '_std'] = float(np.std(self.store[namespace][key]))
                stats[key + '_last'] = float(self.store[namespace][key][-1])
            
            for pattern in additional_stats:
                key1, op, key2 = pattern.split(' ')
                try:
                    key1, op, key2 = pattern.split(' ')
                    if key1 in stats and key2 in stats and op in ops:
                        stats[f"{key1}_{op}_{key2}"] = ops[op](stats[key1], stats[key2])
                except ValueError:
                    continue

            for plot in plots:
                if plot not in self.store[namespace]:
                    continue

                values = self.store[namespace][plot]

                if plot_compress and len(values) > plot_compress:
                    compressed_avg = [np.mean(values[i:i + plot_compress]) for i in range(0, len(values), plot_compress)]
                    plt.plot(range(len(compressed_avg)), compressed_avg, marker='o', linestyle='-')
                    plt.xlabel(f'Episode (x{plot_compress})')
                    plt.title(plot.replace('_', ' ').capitalize())
                    plt.ylabel(plot.split('_')[0].capitalize())
                    plt.savefig(os.path.join(path, f'{plot}_avg.png'))
                    plt.close()

                    compressed_max = [np.max(values[i:i + plot_compress]) for i in range(0, len(values), plot_compress)]
                    plt.plot(range(len(compressed_max)), compressed_max, marker='o', linestyle='-')
                    plt.xlabel(f'Episode (x{plot_compress})')
                    plt.title(plot.replace('_', ' ').capitalize())
                    plt.ylabel(plot.split('_')[0].capitalize())
                    plt.savefig(os.path.join(path, f'{plot}_max.png'))
                    plt.close()
                else:
                    plt.plot(values, linestyle='-')
                    plt.xlabel('Episode')
                    plt.title(plot.replace('_', ' ').capitalize())
                    plt.ylabel(plot.split('_')[0].capitalize())
                    plt.savefig(os.path.join(path, f'{plot}.png'))
                    plt.close()

            summary[namespace] = stats
        
        with open(os.path.join(path, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

