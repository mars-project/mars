import os

output_dir = os.path.join(os.path.dirname(__file__), 'output')

os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, f'{os.getpid()}'), 'w') as f:
    f.write('')
