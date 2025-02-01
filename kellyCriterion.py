import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Kelly Criterion Simulator",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
    
)

st.title("📈 Kelly Criterion Strategy Simulator")
st.markdown("""
### Understanding the Kelly Criterion

The Kelly Criterion is a mathematical framework for optimal bet sizing in scenarios with favorable odds. It answers the question: 
"What fraction of your bankroll should you bet to maximize long-term growth?"

#### Key Concepts:

1. **The Formula**: 
   The Kelly fraction (f*) is calculated as: f* = (bp - q)/b
   where:
   - p is the probability of winning
   - q is the probability of losing (1-p)
   - b is the odds ratio (net winnings divided by potential loss)

2. **Growth Rate**:
   The expected logarithmic growth rate G(f) for a betting fraction f is:
   G(f) = p·ln(1 + bf) + q·ln(1 - f)

3. **Properties**:
   - Maximizes long-term growth rate
   - Provides natural protection against ruin
   - Can be highly volatile in the short term
   - Assumes constant bankroll percentage betting
""")

# Sidebar with more flexible inputs
with st.sidebar:
    st.header("Simulation Parameters")
    
    # Advanced options toggle
    advanced_options = st.checkbox("Show Advanced Options", False)
    
    # Basic parameters with more flexibility
    p = st.slider("Win Probability (p)", 0.01, 0.99, 0.55, 
                  help="Probability of winning each bet")
    
    # Flexible win/loss amounts
    win_amt = st.number_input("Win Amount (% of bet)", 0.1, 1000.0, 100.0, 
                             help="Percentage of bet won on success")
    loss_amt = st.number_input("Loss Amount (% of bet)", 0.1, 100.0, 100.0, 
                              help="Percentage of bet lost on failure")
    
    initial_wealth = st.number_input("Initial Wealth", 
                                   min_value=1, 
                                   max_value=10000000,
                                   value=1000,
                                   format="%d",
                                   help="Starting bankroll amount")
    
    # Advanced parameters
    if advanced_options:
        num_simulations = st.slider("Number of Simulations", 10, 200, 100,
                                  help="Number of parallel simulations to run")
        num_periods = st.slider("Number of Periods", 10, 1000, 100,
                              help="Number of betting periods to simulate")
        custom_fraction = st.checkbox("Use Custom Betting Fraction", False)
        if custom_fraction:
            betting_fraction = st.slider("Custom Betting Fraction", 0.0, 1.0, 0.5,
                                      help="Override Kelly fraction with custom value")
    else:
        num_simulations = 100
        num_periods = 200
        custom_fraction = False
        betting_fraction = None

# Calculate Kelly fraction
b = win_amt / loss_amt
q = 1 - p
kelly_fraction = (b * p - q) / b

if kelly_fraction < 0:
    st.error("⚠️ Negative Kelly fraction - this is a negative expectation bet! The optimal strategy is not to bet at all.")
    st.stop()

# Use custom fraction if specified
fraction_to_use = betting_fraction if custom_fraction else kelly_fraction

# Simulation function
def simulate_kelly(initial, f, p, win_ratio, loss_ratio, periods, simulations):
    results = np.zeros((simulations, periods + 1))
    results[:, 0] = initial
    
    for i in range(simulations):
        for t in range(1, periods + 1):
            if np.random.binomial(1, p):
                results[i, t] = results[i, t-1] * (1 + f * win_ratio)
            else:
                results[i, t] = results[i, t-1] * (1 - f * loss_ratio)
    return results

# Run simulations
sim_results = simulate_kelly(
    initial_wealth, fraction_to_use, p,
    win_amt/100, loss_amt/100,
    num_periods, num_simulations
)

# Calculate statistics
mean_wealth = np.mean(sim_results, axis=0)
std_wealth = np.std(sim_results, axis=0)
log_mean = np.mean(np.log(sim_results), axis=0)
log_std = np.std(np.log(sim_results), axis=0)

# Create subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Linear Scale Trajectories', 'Log Scale Trajectories'))

# Add individual trajectories to linear plot (disable hover)
for i in range(num_simulations):
    fig.add_trace(
        go.Scatter(y=sim_results[i], mode='lines', 
                  line=dict(color='rgba(70, 130, 180, 0.1)'),
                  showlegend=False,
                  hoverinfo='skip'),  # Disable hover
        row=1, col=1
    )

fig.add_trace(
    go.Scatter(y=mean_wealth + std_wealth, mode='lines',
               line=dict(color='rgba(255, 215, 0, 0.2)'), name='+σ',
               showlegend=True,
               hoverinfo='skip'),  # Disable hover
    row=1, col=1
)
fig.add_trace(
    go.Scatter(y=mean_wealth - std_wealth, mode='lines',
               line=dict(color='rgba(255, 215, 0, 0.2)'), 
               fill='tonexty', name='-σ',
               hoverinfo='skip'),  # Disable hover
    row=1, col=1
)
fig.add_trace(
    go.Scatter(y=mean_wealth, mode='lines',
               line=dict(color='gold', width=2), name='Mean',
               hovertemplate='Mean: %{y:.2f}<extra></extra>'),  # Custom hover template
    row=1, col=1
)

for i in range(num_simulations):
    fig.add_trace(
        go.Scatter(y=np.log(sim_results[i]), mode='lines',
                  line=dict(color='rgba(70, 130, 180, 0.1)'),
                  showlegend=False,
                  hoverinfo='skip'),  # Disable hover
        row=1, col=2
    )

# Add mean and standard deviation bands to log plot (disable hover)
fig.add_trace(
    go.Scatter(y=log_mean + log_std, mode='lines',
               line=dict(color='rgba(255, 215, 0, 0.2)'),
               showlegend=False,
               hoverinfo='skip'),  # Disable hover
    row=1, col=2
)
fig.add_trace(
    go.Scatter(y=log_mean - log_std, mode='lines',
               line=dict(color='rgba(255, 215, 0, 0.2)'),
               fill='tonexty', showlegend=False,
               hoverinfo='skip'),  # Disable hover
    row=1, col=2
)
fig.add_trace(
    go.Scatter(y=log_mean, mode='lines',
               line=dict(color='gold', width=2),
               showlegend=False,
               hovertemplate='Log Mean: %{y:.2f}<extra></extra>'),  # Custom hover template
    row=1, col=2
)

# Update layout
fig.update_layout(
    height=500,
    template='plotly_dark',
    showlegend=True,
    hovermode='x unified'
)

fig.update_xaxes(title_text='Period', row=1, col=1)
fig.update_xaxes(title_text='Period', row=1, col=2)
fig.update_yaxes(title_text='Wealth', row=1, col=1)
fig.update_yaxes(title_text='Log Wealth', row=1, col=2)

st.plotly_chart(fig, use_container_width=True)

max_f = min(2, 1/loss_amt*100)
f_values = np.linspace(0, max_f, 500)
growth_rates = [p * np.log(1 + (win_amt/100)*f) + (1-p)*np.log(1 - (loss_amt/100)*f) 
               for f in f_values]

# Convert growth rates to percentages
growth_rates = [r * 100 for r in growth_rates]  # Convert to percentages

# Calculate good axis ranges
x_max = min(max_f, kelly_fraction * 2.5)
x_min = 0
y_min = max(min(growth_rates), -2.5)  # Limit bottom to -2.5%
y_max = max(growth_rates) * 1.1

fig2 = go.Figure()

# Add growth rate curve
fig2.add_trace(go.Scatter(
    x=f_values,
    y=growth_rates,
    mode='lines',
    line=dict(color='rgb(70, 130, 180)', width=2),
    name='Growth Rate'
))

# Add Kelly fraction marker (adjust the y-value since we're now using percentages)
fig2.add_trace(go.Scatter(
    x=[kelly_fraction],
    y=[max(growth_rates)],
    mode='markers',
    marker=dict(color='gold', size=12),
    name=f'Kelly Fraction ({kelly_fraction:.2%})'
))

fig2.update_layout(
    title='Expected Growth Rate vs Betting Fraction',
    xaxis_title='Betting Fraction (f)',
    yaxis_title='Expected Log Growth Rate (%)',
    template='plotly_dark',
    height=500,
    width=600,  # Set explicit width to match subplots
    showlegend=True,
    xaxis_range=[x_min, x_max],
    yaxis_range=[y_min, y_max],
    margin=dict(l=50, r=50, t=50, b=50)
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.plotly_chart(fig2, use_container_width=True)