import json

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

    def save(self, filename):
        with open(filename + '.json', 'w') as f:
            json.dump(self.store, f)

    def summary(self, filename, plots=[], additional_stats=[]):
        summary = {}
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
                stats[key1 + '_' + op + '_' + key2] = eval(f'{stats[key1]} {op} {stats[key2]}')

            for plot in plots:
                plt.plot(self.store[namespace][plot])
                plt.title(plot.replace('_', ' ').capitalize())
                plt.xlabel('Episode')
                plt.ylabel(plot.split('_')[0].capitalize())
                plt.savefig(filename + f'_{plot}.png')
                plt.close()

            summary[namespace] = stats
        
        with open(filename + '_summary.json', 'w') as f:
            json.dump(summary, f)

