import json
import pandas as pd

with open('path/to/file/that/you/want/to/test_factool_answers1.json', 'r') as file:
    data = json.load(file)
df = pd.DataFrame(data)
print(len(data))
average_claim_level_factuality_mean = df['average_claim_level_factuality'].mean()
average_response_level_factuality_mean = df['average_response_level_factuality'].mean()

print(f"Average of average_claim_level_factuality: {average_claim_level_factuality_mean}")
print(f"Average of average_response_level_factuality: {average_response_level_factuality_mean}")




