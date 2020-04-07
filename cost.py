def range_fix(a, b, value, max_value, min_value):
    new_value = ((b-a) * (max_value - value)) / (max_value - min_value) + a
    return new_value

def crit_weights_gen(a, b, class_no):
    max_example = max(class_no)
    min_example = min(class_no)
    class_weight = []
    max_example_index = class_no.index(max_example)
    for i in range(len(class_no)):
        if i is not max_example_index:
            class_weight.append((class_no[i]/max_example)*10)
    max_value = max(class_weight)
    min_value = min(class_weight)
    class_weight = [range_fix(a, b, class_weight[i], max_value, min_value) for i in range(len(class_weight)) if class_weight[i]]
    class_weight = class_weight[:max_example_index] + [a] + class_weight[max_example_index:]
    return class_weight