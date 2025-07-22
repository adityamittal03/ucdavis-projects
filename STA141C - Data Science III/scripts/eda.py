import matplotlib.pyplot as plt
import seaborn as sns
from .utils import load_data


def main():
    _, _, df = load_data(sample=10000, standardize=False)

    flights_per_airline = df.groupby('Airline').agg({'Class': 'count'})
    delays_by_airline = df.groupby('Airline').agg({'Class': 'sum'})
    flights_per_airline['delays'] = delays_by_airline['Class']
    flights_per_airline['%_delays'] = flights_per_airline['delays'] / flights_per_airline['Class']

    plt.figure(figsize=(10,7))
    sns.countplot(y='Airline', hue='Class', data=df)
    plt.xlabel('Number of Flights')
    plt.title('Number of Flights On Time and Delayed by Airline')
    plt.legend(labels=['Delayed', 'On-time'])
    plt.tight_layout()
    plt.savefig('flights_by_airline.png')

    plt.figure(figsize=(8,6))
    sns.barplot(y='Airline', x='%_delays', data=flights_per_airline, orient='h', color='C0')
    plt.xlabel('Percent of Flights Delayed')
    plt.title('Percent of Delays by Airline')
    plt.tight_layout()
    plt.savefig('percent_delays_airline.png')

    delayed = df[df['Class'] == 1]
    plt.figure(figsize=(8,5))
    plt.hist([df['Length'], delayed['Length']], label=['All', 'Delayed'], density=True)
    plt.xlabel('Length of Flight')
    plt.title('Normalized Histogram of Length of Flight')
    plt.legend()
    plt.tight_layout()
    plt.savefig('length_hist.png')

    plt.figure(figsize=(8,5))
    plt.hist([df['DayOfWeek'], delayed['DayOfWeek']], label=['All', 'Delayed'], density=True)
    plt.xlabel('Day of Week')
    plt.title('Normalized Histogram of Day of Week')
    plt.legend()
    plt.tight_layout()
    plt.savefig('day_hist.png')

    print('EDA plots saved as PNG files.')


if __name__ == "__main__":
    main()
