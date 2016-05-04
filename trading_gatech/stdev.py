"""Implementation of Standard Deviation"""

import math as m

# calculate the population standard deviation
def stdev_p(data):
    mean = sum(data) / len(data)
    sum_dev = 0.
    for x in data:
        sum_dev += (x - avg)**2
    return m.sqrt(sum_dev / len(data))

# calculate the sample standard deviation
def stdev_s(data):
    mean = sum(data) / len(data)
    sum_dev = 0.
    for x in data:
        sum_dev += (x - avg)**2
    return m.sqrt(sum_dev / (len(data) - 1))

if __name__ == "__main__":
    test = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    print "the population stdev is", stdev_p(test)
    print "the sample stdev is", stdev_s(test)
