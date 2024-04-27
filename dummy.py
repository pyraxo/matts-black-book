import csv

data = [
    {"Car": "Toyota", "Review": "Great car", "Rating": 5},
    {"Car": "Honda", "Review": "Reliable and fuel-efficient", "Rating": 4},
    {"Car": "Ford", "Review": "Powerful engine", "Rating": 4.5},
    {"Car": "Chevrolet", "Review": "Comfortable ride", "Rating": 3.5},
]

filename = "/Users/aaron/Projects/matts-black-book/data/sample.csv"

with open(filename, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["Car", "Review", "Rating"])
    writer.writeheader()
    writer.writerows(data)

print(f"Sample CSV file '{filename}' has been generated.")
