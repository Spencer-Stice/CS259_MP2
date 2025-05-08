def find_inputs_with_multiple_x(min_x=4, max_x=20):
    ranges = {
        'small': range(16, 65),
        'medium': range(65, 161),
        'large': range(300, 400)
    }

    results = {}

    for size_label, input_range in ranges.items():
        for n in input_range:
            valid_x = []
            for x in range(min_x, max_x + 1):
                if x > 2 and n % x == 0 and (n - 2) % (x - 2) == 0:
                    valid_x.append(x)
            if len(valid_x) >= 3:
                results[size_label] = (n, valid_x[:3])
                break  # just need one for each range
    return results

# Run and print
result = find_inputs_with_multiple_x()
for size in ['small', 'medium', 'large']:
    n, xs = result[size]
    print(f"{size.capitalize()} input: {n}, valid x values: {xs}")
