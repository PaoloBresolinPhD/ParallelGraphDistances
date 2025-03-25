import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # load the dataframe with data
    file_name = sys.argv[1]
    results_df = pd.read_csv(file_name)

    # create and save a plot with the wallclock times w.r.t. the number of processes
    sns.set_theme()
    sns.set_style("whitegrid")

    results_df['Elapsed'] = pd.to_timedelta(results_df['Elapsed']).dt.total_seconds() / 60

    sns.barplot(data=results_df, x='NCores', y='Elapsed')
    plt.xlabel('Number of Cores')
    plt.ylabel('Elapsed Time (min)')
    plt.title('Elapsed Time vs Number of Cores')

    plt.savefig('./plots/wallclock_time.png')
    plt.close()

    # create and save a plot with the maximum RAM usage w.r.t. the number of processes
    sns.set_theme()
    sns.set_style("whitegrid")

    results_df['MaxRSS'] = results_df['MaxRSS'].str.replace("K", "", regex=True).astype(float) / 1_048_576  # convert from KB to GB

    sns.barplot(data=results_df, x='NCores', y='MaxRSS')
    plt.xlabel('Number of Cores')
    plt.ylabel('Max RAM (GB)')
    plt.title('RAM Usage vs Number of Cores')

    plt.savefig('./plots/max_ram.png')
    plt.close()