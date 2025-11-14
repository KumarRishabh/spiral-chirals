import numpy as np
import plotly.graph_objects as go

# Number of samples along the curve
n = 2000

# Parameter t from 0 to 10π
t = np.linspace(0, 10*np.pi, n)

# Archimedean spiral: r = a + b t
a = 0.0
b = 0.1
r_arch = a + b*t
x_arch = r_arch*np.cos(t)
y_arch = r_arch*np.sin(t)

# Fermat spiral: r = c * sqrt(t)
c = 0.3
r_fermat = c*np.sqrt(t)
x_fermat = r_fermat*np.cos(t)
y_fermat = r_fermat*np.sin(t)

# Build Plotly figure with both spirals
# Create separate figures for each spiral type

# Archimedean Spiral Figure
fig_arch = go.Figure()
fig_arch.add_trace(go.Scatter(x=x_arch, y=y_arch,
                             mode='lines',
                             name='Archimedean',
                             line=dict(color='blue')))

fig_arch.update_yaxes(scaleanchor="x", scaleratio=1)
fig_arch.update_layout(
    title='Archimedean Spiral (r = a + bt)<br>Equal spacing between successive turns',
    xaxis_title='x',
    yaxis_title='y',
    showlegend=False
)
fig_arch.write_image('archimedean_spiral.svg')

# Fermat Spiral Figure  
fig_fermat = go.Figure()
fig_fermat.add_trace(go.Scatter(x=x_fermat, y=y_fermat,
                               mode='lines',
                               name='Fermat',
                               line=dict(color='red')))

fig_fermat.update_yaxes(scaleanchor="x", scaleratio=1)
fig_fermat.update_layout(
    title='Fermat Spiral (r = c√t)<br>Also known as parabolic spiral',
    xaxis_title='x',
    yaxis_title='y',
    showlegend=False
)
fig_fermat.write_image('fermat_spiral.svg')