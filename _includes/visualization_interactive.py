import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib_venn import venn3
from plotly.subplots import make_subplots
from pathlib import Path
from plotly.colors import sample_colorscale
from matplotlib import cm, colors as mcolors
import itertools
from src.scripts.implementations import *
import seaborn as sns

def barplot_subgraph_feasibility_parallel_plotly(df_html, GENERATED_DATA_FOLDER, update=False, data_folder=None):
    """Bar plot to compare the number of paths when restricting the graph using Plotly."""
    save_folder = GENERATED_DATA_FOLDER / "shortest_path_matrices/"
    save_folder.mkdir(parents=True, exist_ok=True)

    if not update:
        try:
            full_dist_matrix = np.load(save_folder / "full_dist_matrix.npz")['arr_0']
            lead_dist_matrix = np.load(save_folder / "lead_dist_matrix.npz")['arr_0']
            infobox_dist_matrix = np.load(save_folder / "infobox_dist_matrix.npz")['arr_0']
            first_dist_matrix = np.load(save_folder / "first_dist_matrix.npz")['arr_0']
            print("Loaded precomputed distance matrices from disk.")
        except (FileNotFoundError, KeyError):
            print("Precomputed distance matrices not found. Recomputing all matrices...")
            update = True

    if update:
        # Full graph
        print("Computing full graph shortest path matrix...")
        full_dist_matrix, articles_list = shortest_path_matrix_parallel(df_html, pd.Series([True]*len(df_html)))

        # Lead section
        print("Computing lead section shortest path matrix...")
        lead_dist_matrix, _ = shortest_path_matrix_parallel(df_html, df_html.section_category == 'lead')

        # Infobox
        print("Computing infobox section shortest path matrix...")
        infobox_dist_matrix, _ = shortest_path_matrix_parallel(df_html, df_html.section_category == 'infobox')

        # First link
        print("Computing first link shortest path matrix...")
        first_dist_matrix, _ = shortest_path_matrix_parallel(df_html, df_html.index == 1)

        # Save
        np.savez_compressed(save_folder / "full_dist_matrix.npz", full_dist_matrix)
        np.savez_compressed(save_folder / "lead_dist_matrix.npz", lead_dist_matrix)
        np.savez_compressed(save_folder / "infobox_dist_matrix.npz", infobox_dist_matrix)
        np.savez_compressed(save_folder / "first_dist_matrix.npz", first_dist_matrix)

    # Info for the barplot:
    labels = ['Full graph', 'Lead only', 'Infobox only', 'First link only']
    counts = [
        np.isfinite(full_dist_matrix).sum(),
        np.isfinite(lead_dist_matrix).sum(),
        np.isfinite(infobox_dist_matrix).sum(),
        np.isfinite(first_dist_matrix).sum()
    ]

    df = pd.DataFrame({
        'Graph Restriction': labels,
        'Number of Paths': counts
    })

    # Format hover text with apostrophe thousands separator
    hover_texts = [f"{cnt:,}".replace(",", "'") for cnt in counts]

    fig = px.bar(
        df,
        x='Graph Restriction',
        y='Number of Paths',
        color='Graph Restriction',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        log_y=True,
        title="Comparison of path possibilities under different graph restrictions",
        hover_data={'Number of Paths': False},  # hide default hover
        text=hover_texts  # show formatted numbers on bars
    )

    # Set custom hover template
    fig.update_traces(
        hovertemplate='Count: %{text}',  # only show formatted number
        textposition='outside'
    )

    # Layout
    fig.update_layout(
        showlegend=False,
        template="plotly_white",
        title_x=0.5
    )

    fig.show()

    if data_folder is not None:
        fig.write_html(
            data_folder / "barplot_subgraph_feasibility_parallel_plotly.html",
            full_html=False,
            include_plotlyjs='cdn'
        )

def upsetplot_section_categories_plotly(df_html, data_folder= None):

    lead_series = (df_html.section_category == 'lead')*1
    infobox_series = (df_html.section_category == 'infobox')*1
    body_series = (df_html.section_category == 'body')*1
    df = pd.DataFrame({'articles': df_html.linkTarget, 'Lead': lead_series, 'Infobox': infobox_series, 'Body': body_series})
    df = df.groupby('articles').aggregate({'Lead':'max', 'Infobox':'max', 'Body':'max'}).reset_index()
    df.drop(columns = ['articles'], inplace = True)

    # CREDITS: https://community.plotly.com/t/plotly-upset-plot/63858
    # an array of dimensions d x d*2^d possible subsets where d is the number of columns
    subsets = []
    # the sizes of each subset (2^d array)
    subset_sizes = [ ]
    d = len(df.columns)
    for i in range(1, d + 1):
        subsets = subsets + [list(x) for x in list(itertools.combinations(df.columns, i))]
        
    for s in subsets:
        curr_bool = [1]*len(df)
        for col in df.columns:
            if col in s: curr_bool = [x and y for x, y in zip(curr_bool, list(df.loc[:, col].copy()))]
            else: curr_bool = [x and not y for x, y in zip(curr_bool, list(df.loc[:, col].copy()))]
        subset_sizes.append(sum(curr_bool))
    
    
    plot_df = pd.DataFrame({'Intersection': subsets, 'Size':subset_sizes})
    plot_df = plot_df.sort_values(by = 'Size', ascending = False)
    max_y = max(plot_df['Size'])+0.1*max(plot_df['Size'])
    
    subsets = list(plot_df['Intersection'])
    scatter_x = []
    scatter_y = []
    for i, s in enumerate(subsets):
        for j in range(d):
            scatter_x.append(i)
            scatter_y.append(-j*max_y/d-0.1*max_y)
            
    fig = go.Figure()
#     fig.add_trace(go.Scatter(x=[-1.2,len(subsets)],y= [max_y+0.1*max_y,max_y+0.1*max_y],fill='tozeroy'))
    template =  ['' for x in scatter_x]
    fig.add_trace(go.Scatter(x = scatter_x, y = scatter_y, mode = 'markers', showlegend=False, marker=dict(size=16,color='#C9C9C9'), hovertemplate = template))
    fig.update_layout(xaxis=dict(showgrid=False, zeroline=False),
                  yaxis=dict(showgrid=True, zeroline=False),
                   plot_bgcolor = "#FFFFFF", margin=dict(t=40, l=150)) 
    
    for i, s in enumerate(subsets):
        scatter_x_has = []
        scatter_y_has = []
        for j in range(d):
            if df.columns[j] in s:
                scatter_x_has.append(i)
                scatter_y_has.append(-j*max_y/d-0.1*max_y)
                fig.add_trace(go.Scatter(x = scatter_x_has, y = scatter_y_has, mode = 'markers+lines', showlegend=False, marker=dict(size=16,color='#000000',showscale=False), hovertemplate = template))
    fig.update_xaxes(showticklabels=False) # Hide x axis ticks 
    fig.update_yaxes(showticklabels=False) # Hide y axis ticks
    fig.update_traces(hoverinfo=None)
    
    plot_df['Intersection'] = ['+'.join(x) for x in plot_df['Intersection']]
    template =  [f'<extra><br><b>{lab}</b><br><b>N-Count</b>: {n}</extra>' for  lab, n in zip(plot_df['Intersection'], plot_df['Size'])]
    bar = go.Bar(x = list(range(len(subsets))), y = plot_df['Size'], marker = dict(color='#000000'),  text = plot_df['Size'], hovertemplate = template, textposition='outside', hoverinfo='none')
    fig.add_trace(bar)
    
    template =  ['' for x in range(d)]
    max_string_len = max([len(x) for x in df.columns])
    fig_lab = go.Scatter(x = [-0.01*max_string_len]*d, y = scatter_y, text = df.columns, mode = 'text', textposition='middle left',showlegend=False, hovertemplate = template)
    fig_lab = go.Scatter(x = [-0.01*max_string_len]*d, y = scatter_y, text = df.columns, mode = 'text', textposition='middle left',showlegend=False, hovertemplate = template)
    fig.add_trace(fig_lab)
    fig.update_layout(title = '<b>Intersections<b>', yaxis_range=[-max_y-0.1*max_y-1,max_y+0.1*max_y], xaxis_range = [-0.13*max_string_len, len(subsets)], showlegend = False, title_x=0.5)
    
    if data_folder is not None:
        fig.write_html(
            data_folder / f"upsetplot_section_category.html", 
            full_html=False,
            include_plotlyjs='cdn' 
        )
    return fig

def plot_category_boxplots_plotly(df, value_col, n_categories=10, data_folder= None):
    """
    Boxplots of a value across a random sample of categories using Plotly.
    """
    # Data Preparation
    df_exp = df.explode("category").dropna(subset=["category"])
    df_exp = df_exp.drop_duplicates(["article", "category"])
    categories = df_exp["category"].unique()
    
    if n_categories < len(categories):
        # Use numpy random choice
        categories = np.random.choice(categories, size=n_categories, replace=False)
    
    df_exp = df_exp[df_exp["category"].isin(categories)]

    fig = px.box(
        df_exp, 
        x="category", 
        y=value_col,
        title=f"Boxplots of {value_col} by Category",
        points="outliers"
    )
    
    fig.update_layout(
        xaxis_title="Category",
        yaxis_title=value_col,
        template="plotly_white",
        height=600,
        width=1000
    )
    
    fig.write_html(
        data_folder / f"boxplots_{value_col}_by_category.html", 
        full_html=False,
        include_plotlyjs='cdn' 
    )
    return fig

def barplot_first_n_links_plotly(df_html, GENERATED_DATA_FOLDER, n_start=1, max_n=20, step=1, update=False):
    """
    Plot the number of finite paths using first n hyperlinks,
    plus the full graph, with a gradient color scale.
    The 'Full' bar has a custom hover text.
    """

    save_folder = Path(GENERATED_DATA_FOLDER) / "shortest_path_first_n"
    save_folder.mkdir(parents=True, exist_ok=True)

    finite_counts = []
    available_matrices = []

    ns = list(range(n_start, max_n + 1, step))

    # Load or compute matrices
    if not update:
        all_loaded = True
        for n in ns:
            path = save_folder / f"first_{n}_links_matrix.npz"
            try:
                available_matrices.append(np.load(path)["arr_0"])
            except (FileNotFoundError, KeyError):
                all_loaded = False
                break
        if not all_loaded:
            update = True

    if update:
        available_matrices = []
        for n in ns:
            condition = df_html["index"] <= n
            dist_matrix, _ = shortest_path_matrix_parallel(df_html, condition)
            available_matrices.append(dist_matrix)
            np.savez_compressed(save_folder / f"first_{n}_links_matrix.npz", dist_matrix)

    # Count paths
    for matrix in available_matrices:
        finite_counts.append(np.isfinite(matrix).sum())

    # Add full graph
    full_matrix_path = Path(GENERATED_DATA_FOLDER) / "shortest_path_matrices/full_dist_matrix.npz"
    full_dist_matrix = np.load(full_matrix_path)["arr_0"]
    full_count = np.isfinite(full_dist_matrix).sum()
    finite_counts.append(full_count)

    # Labels as strings
    labels = [str(n) for n in ns] + ["Full"]

    # Gradient colors
    colorscale = px.colors.sequential.Emrld
    color_values = list(ns) + [max(ns)]
    bar_colors = sample_colorscale(colorscale, [i / max(color_values) for i in color_values])

    # Hover text
    hover_texts = [f"First {n} links<br>Number of paths: {count:,}" for n, count in zip(ns, finite_counts[:-1])]
    hover_texts.append(f"Full graph<br>Number of paths: {full_count:,}")  # last bar

    # Plotly bar chart
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=finite_counts,
            marker_color=bar_colors,
            hovertext=hover_texts,
            hoverinfo="text"
        )
    )

    fig.update_layout(
        title="Graph Reachability Using First n Hyperlinks",
        xaxis_title="First n hyperlinks",
        yaxis_title="Number of finite paths",
        template="plotly_white",
        bargap=0.2,
        height=500,
        width=900
    )

    fig.show()

def plot_stacked_path_length_distribution_plotly(df, GENERATED_DATA_FOLDER, data_folder= None, k=5, base_cmap="Blues"):
    """
    Interactive stacked bar plot of path length distributions with gradient for lengths < k
    and a single stabilized color for lengths >= k. Adds a "Full" bar at the end
    with custom hover text.
    
    Parameters:
    - df: DataFrame from compute_shortest_length_counts_per_n (indexed by n)
    - GENERATED_DATA_FOLDER: folder containing full_dist_matrix.npz
    - k: path lengths < k get gradient colors, lengths >= k get single color
    - base_cmap: matplotlib colormap
    """

    if df.empty:
        print("Empty DataFrame — nothing to plot.")
        return

    full_matrix_path = Path(GENERATED_DATA_FOLDER) / "shortest_path_matrices/full_dist_matrix.npz"
    full_dist_matrix = np.load(full_matrix_path)["arr_0"]
    full_count_matrix = np.rint(full_dist_matrix[np.isfinite(full_dist_matrix)]).astype(int)

    # Compute counts per path length
    max_length = max(df.columns.str.replace("len_", "").astype(int).max(),
                     full_count_matrix.max())
    full_counts_dict = {f"len_{l}": 0 for l in range(1, max_length+1)}
    for length, count in pd.Series(full_count_matrix).value_counts().items():
        if length == 0:
            continue
        full_counts_dict[f"len_{length}"] = count

    # Append full as last row
    df_full = pd.concat([df, pd.DataFrame(full_counts_dict, index=["Full"])])

    # Colors
    cmap = cm.get_cmap(base_cmap)
    num_cols = len(df_full.columns)
    gradient_colors = [mcolors.to_hex(cmap((i+1)/k)) for i in range(k-1)]
    stabilized_color = mcolors.to_hex(cmap(k/(k+1)))
    colors = []
    for idx in range(num_cols):
        length = int(df_full.columns[idx].replace("len_", ""))
        if length < k:
            colors.append(gradient_colors[length-1])
        else:
            colors.append(stabilized_color)

    # Legend
    legend_labels = []
    for idx, col in enumerate(df_full.columns):
        length = int(col.replace("len_", ""))
        if length < k:
            legend_labels.append(str(length))
        else:
            legend_labels.append(f"length ≥ {k}")

    # Build bars
    seen_labels = set()
    bars = []
    bottom = np.zeros(len(df_full))
    x_labels = df_full.index.astype(str)

    for i, col in enumerate(df_full.columns):
        label = legend_labels[i]
        show_legend = label not in seen_labels
        seen_labels.add(label)

        # Hover
        hover_texts = []
        for j, x_val in enumerate(x_labels):
            if x_val == "Full":
                hover_texts.append(f"Full graph<br>Path length {label}: {df_full[col].iloc[j]:,}")
            else:
                hover_texts.append(f"First n links: {x_val}<br>Path length {label}: {df_full[col].iloc[j]:,}")

        bars.append(
            go.Bar(
                x=x_labels,
                y=df_full[col].values,
                name=label,
                marker_color=colors[i],
                offsetgroup=0,
                base=bottom,
                hovertext=hover_texts,
                hoverinfo="text",
                showlegend=show_legend,
                marker_line_width=0
            )
        )
        bottom += df_full[col].values

    # Plot
    fig = go.Figure(data=bars)
    fig.update_layout(
        barmode='stack',
        title="Shortest Path Length Distribution",
        xaxis_title="First n hyperlinks",
        yaxis_title="Count of ordered pairs",
        template="plotly_white",
        height=500,
        width=900,
        bargroupgap=0
    )
    fig.write_html(
        data_folder / f"plot_stacked_path_length_distribution_plotly.html", 
        full_html=False,
        include_plotlyjs='cdn' 
    )
    fig.show()

def plot_links_sections_plotly(df, data_folder= None):
    """
    Bar chart of link counts by section number using Plotly.
    """
    # Data Preparation
    counts = df["section_number"].value_counts().sort_index()
    
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        labels={'x': 'Section number', 'y': 'Count'},
        title="Placement of links in page sections"
    )
    
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(tickmode='linear'), # Ensure all integers show
        height=600,
        width=1000
    )
    

    fig.write_html(
        data_folder / f"links_sections.html", 
        full_html=False,
        include_plotlyjs='cdn' 
    )
    return fig


def plot_hyperlink_section_donut_plotly(df, data_folder= None):
    """
    Donut chart of hyperlink distribution by section category using Plotly.
    Note: 'ax' argument is removed as it is specific to Matplotlib.
    """
    if "section_category" not in df.columns:
        raise ValueError("DataFrame must contain a 'section_category' column.")
    
    counts = df["section_category"].dropna().value_counts().sort_values(ascending=False)
    
    if counts.empty:
        raise ValueError("No data available to plot: 'section_category' column is empty.")
    color_map = {
        "infobox": "#A1C9F4",
        "lead": "#FFB482",
        "body": "#8DE5A1"
    }
    # Create donut chart
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        color=counts.index,
        hole=0.35, # Creates the donut look
        title="Hyperlink distribution by section category",
        color_discrete_map=color_map,
    )
    
    fig.update_traces(
        textinfo='percent+label',
        textposition='outside',
        pull=[0.02] * len(counts),
        hovertemplate=[
            f"Count: {cnt:,}".replace(",", "'") for cnt in counts.values
        ]
    )

    fig.update_layout(
        template="plotly_white",
        height=500,
        width=600
    )
    
    fig.write_html(
        data_folder / f"hyperlink_section_donut.html", 
        full_html=False,
        include_plotlyjs='cdn'
    )
    return fig


def plot_top_10_link_targets_plotly(df, data_folder= None):
    """
    Horizontal bar chart of the top 10 most linked targets using Plotly.
    """
    counts = df["linkTarget"].value_counts().head(10)
    
    fig = px.bar(
        x=counts.values,
        y=counts.index,
        orientation='h',
        title="Top links by frequency",
        labels={'x': 'Count', 'y': 'Link target'}
    )
    
    # Invert y-axis to have top item at top
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        height=600,
        width=1000
    )
    
    fig.write_html(
        data_folder / f"top_10_link_targets.html", 
        full_html=False,
        include_plotlyjs='cdn' 
    )
    return fig


def plot_link_target_source_jointplot_plotly(df, data_folder=None):
    """
    Joint plot (scatter with marginal histograms) of link targets vs their sources,
    with clean hover texts.
    """
    linkTarget_counts = df["linkTarget"].value_counts()
    linkSource_counts = df["linkSource"].value_counts()

    # Align data by index (Page Name)
    combined_df = pd.DataFrame({
        'Source Counts': linkSource_counts,
        'Target Counts': linkTarget_counts
    }).dropna()  # Only pages present in both

    # Create scatter with marginal histograms
    fig = px.scatter(
        combined_df,
        x="Source Counts",
        y="Target Counts",
        marginal_x="histogram",
        marginal_y="histogram",
        title="Joint Plot of Link Targets vs Their Sources",
        opacity=0.6
    )

    # Clean hover text
    hover_texts = [
        f"Page: {name}<br>Source links: {row['Source Counts']:,}<br>Target links: {row['Target Counts']:,}"
        for name, row in combined_df.iterrows()
    ]
    fig.update_traces(hovertext=hover_texts, hoverinfo="text")

    # Annotate Top Targets
    top_targets = combined_df.nlargest(3, 'Target Counts')
    for name, row in top_targets.iterrows():
        fig.add_annotation(
            x=row['Source Counts'],
            y=row['Target Counts'],
            text=name,
            yshift=10,
        )

    # Annotate Top Sources, avoiding duplicates
    top_sources = combined_df.nlargest(3, 'Source Counts')
    for name, row in top_sources.iterrows():
        if name not in top_targets.index:
            fig.add_annotation(
                x=row['Source Counts'],
                y=row['Target Counts'],
                text=name,
                yshift=10,
            )

    fig.update_layout(
        template="plotly_white",
        height=700,
        width=800
    )

    # Save HTML if folder provided
    if data_folder is not None:
        fig.write_html(
            data_folder / "link_target_source_jointplot.html",
            full_html=False,
            include_plotlyjs='cdn'
        )

    return fig

def plot_avg_position_hyperlinks_plotly(
    avg_positions_hyperlinks_df, avg_positions_df, n_bins=30, data_folder=None
):
    """
    Histograms (log-y) of average hyperlink positions.
    Returns a subplot figure containing both charts.
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "Distribution of hyperlinks amongst other hyperlinks", 
            "Distribution of the number of hyperlinks"
        )
    )

    # First Histogram
    fig.add_trace(
        go.Histogram(
            x=avg_positions_hyperlinks_df,
            nbinsx=n_bins,
            name="Avg Pos Hyperlinks",
            marker_color='#636EFA'
        ),
        row=1, col=1
    )

    # Second Histogram
    fig.add_trace(
        go.Histogram(
            x=avg_positions_df,
            nbinsx=n_bins,
            name="Avg Pos Overall",
            marker_color='#EF553B'
        ),
        row=2, col=1
    )

    # Update Layout to match log scale and labels
    fig.update_yaxes(type="log", title_text="Frequency", row=1, col=1)
    fig.update_yaxes(type="log", title_text="Frequency", row=2, col=1)
    
    fig.update_xaxes(title_text="Average position", row=1, col=1)
    fig.update_xaxes(title_text="Average position", row=2, col=1)

    fig.update_layout(
        title_text="Hyperlink Position Distributions",
        template="plotly_white",
        showlegend=False,
        height=800,
        width=1000,
        bargap=0.1
    )

    fig.write_html(
        data_folder / f"hyperlink_position_distributions.html", 
        full_html=False,
        include_plotlyjs='cdn' 
    )
    return fig
def plot_relative_position_distribution_plotly(df_html, n_bins=30, data_folder=None):
    color_map = {
        "infobox": "#A1C9F4",
        "lead": "#FFB482",
        "body": "#8DE5A1"
    }
    stack_order = ["body", "lead", "infobox"]
    fig = px.histogram(
        df_html,
        x="relative_position_link",
        color="section_category",
        nbins=n_bins,
        title="Links distribution",
        labels={"relative_position_link": "Average position"},
        color_discrete_map=color_map, # Same colors as seaborn because it is nice
        category_orders={"section_category": stack_order},# Nicer visual order
        template="simple_white" 
    )
    # Add median
    median = df_html["relative_position_link"].median()
    fig.add_vline(
        x=median, 
        line_width=3, 
        line_dash="dash", 
        line_color="black",
        annotation_text=f"Median: {median:.1f}", 
        annotation_position="top right"
    )
    fig.write_html(
        data_folder / f"plot_relative_position_distribution_plotly.html", 
        full_html=False,
        include_plotlyjs='cdn' 
    )
    fig.show()

def plot_absolute_position_distribution_plotly(df_html, n_bins=30, data_folder=None):
    color_map = {
        "infobox": "#A1C9F4",
        "lead": "#FFB482",
        "body": "#8DE5A1"
    }
    stack_order = ["body", "lead", "infobox"]
    fig = px.histogram(
        df_html,
        x="absolute_position_link",
        color="section_category",
        nbins=n_bins,
        title="Links distribution",
        labels={"absolute_position_link": "Average position"},
        color_discrete_map=color_map, # Same colors as seaborn because it is nice
        category_orders={"section_category": stack_order},# Nicer visual order
        template="simple_white" 
    )
    # Add median
    median = df_html["absolute_position_link"].median()
    fig.add_vline(
        x=median, 
        line_width=3, 
        line_dash="dash", 
        line_color="black",
        annotation_text=f"Median: {median:.1f}", 
        annotation_position="top right"
    )
    fig.write_html(
        data_folder / f"plot_absolute_position_distribution_plotly.html", 
        full_html=False,
        include_plotlyjs='cdn' 
    )
    fig.show()



def plot_articles_click_plotly(df_articles, n_bins):
    """
    Interactive histogram (log-y) of number of clicks per article.
    Hover shows: range of clicks and count of articles.
    """
    clicks = df_articles["number_of_clicks"].values

    # Compute histogram
    counts, bin_edges = np.histogram(clicks, bins=n_bins)

    # Midpoints of bins for hover
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Build hover texts
    hover_texts = [
        f"Clicks range: {int(bin_edges[i])} - {int(bin_edges[i+1])}<br>Articles: {counts[i]:,}"
        for i in range(len(counts))
    ]

    fig = go.Figure(
        go.Bar(
            x=bin_midpoints,
            y=counts,
            hovertext=hover_texts,  # hover info only
            hoverinfo="text",
            marker_color='steelblue',
        )
    )

    fig.update_layout(
        title="Distribution of Number of Clicks per Article",
        xaxis_title="Number of clicks",
        yaxis_title="Frequency",
        yaxis_type="log",  # log scale
        template="plotly_white",
        bargap=0.1,
        height=500,
        width=900
    )

    fig.show()

def plot_HITS_plotly(data_folder=None):
    # hard coded results from HITS algorithm
    hubs_data = [
        ('Driving on the left or right', 0.00132), 
        ('Lebanon', 0.00121), 
        ('List of countries', 0.00121), 
        ('List of circulating currencies', 0.00120), 
        ('List of sovereign states', 0.00117), 
        ('List of countries by system of government', 0.00117), 
        ('Georgia (country)', 0.00116), 
        ('Armenia', 0.00114), 
        ('Turkey', 0.00114), 
        ('United States', 0.00113)
    ]

    auth_data = [
        ('Wikipedia Text of the GNU Free Documentation License', 0.0247), 
        ('United States', 0.01171), 
        ('France', 0.00846), 
        ('United Kingdom', 0.00822), 
        ('Europe', 0.00760), 
        ('Germany', 0.00674), 
        ('World War II', 0.00632), 
        ('India', 0.00544), 
        ('Spain', 0.00529), 
        ('Italy', 0.00528)
    ]

    h_names, h_scores = zip(*hubs_data[::-1])
    a_names, a_scores = zip(*auth_data[::-1])

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("<b>Top 10 Hubs</b>", "<b>Top 10 Authorities</b>"),
        horizontal_spacing=0.25
    )

    # Hubs
    fig.add_trace(
        go.Bar(
            y=h_names,
            x=h_scores,
            orientation='h',
            name="Hub Score",
            marker=dict(color='#636EFA'), 
            hovertemplate='<b>%{y}</b><br>Score: %{x:.6f}<extra></extra>'
        ),
        row=1, col=1
    )

    #Authorities
    fig.add_trace(
        go.Bar(
            y=a_names,
            x=a_scores,
            orientation='h',
            name="Authority Score",
            marker=dict(color='#EF553B'),
            hovertemplate='<b>%{y}</b><br>Score: %{x:.6f}<extra></extra>'
        ),
        row=1, col=2
    )

    fig.update_layout(
        title_text="<b>HITS Algorithm Analysis</b>: Top Hubs & Authorities",
        showlegend=False,
        height=600,
        margin=dict(l=20, r=20, t=80, b=20)
    )

    fig.write_html(
        data_folder / f"HITS_plot.html", 
        full_html=False,
        include_plotlyjs='cdn' 
    )
    return fig

def barplot_nunique_in_first_links_plotly(df_html, data_folder=None):
    df = df_html.copy()
    
    # Keep only the first links (deduplication based on index priority)
    df = df.sort_values(by=['index'], ascending=True)
    df.drop_duplicates(subset=['linkTarget'], keep='first', inplace=True)

    # Get first 1-34 links
    subset = df[df['index'].between(1, 34)]
    counts = subset.groupby('index')['linkTarget'].nunique()
    counts = counts.reindex(range(1, 35), fill_value=0)
    
    # Get count for >34
    gt_34_count = df[df['index'] > 34]['linkTarget'].nunique()
    
    # Create the main Series
    counts.index = counts.index.astype(str)
    counts['>34'] = gt_34_count
    
    new_links = counts
    previous_cumsum = counts.cumsum().shift(1).fillna(0)

    # plot
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Links already there',
        x=new_links.index,
        y=previous_cumsum,
        marker_color='#B0B0B0',  # Grey
        marker_line_color='black',
        marker_line_width=1
    ))

    fig.add_trace(go.Bar(
        name='New Links Added',
        x=new_links.index,
        y=new_links,
        marker_color='#1f77b4',  # Blue
        marker_line_color='black',
        marker_line_width=1
    ))

    # Layout configuration
    fig.update_layout(
        barmode='stack',
        title="Cumulative Growth of Unique Links",
        xaxis_title="n first links",
        yaxis_title="Cumulative Count",
        xaxis=dict(
            type='category',
            tickangle=-45
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=1,
            griddash='dash'
        ),
        plot_bgcolor='white',
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='Black',
            borderwidth=1
        ),
        width=800, # Equivalent to figsize (12, 6)
        height=400
    )
    fig.write_html(
        data_folder / f"barplot_nunique_in_first_links_plotly.html", 
        full_html=False,
        include_plotlyjs='cdn' 
    )
    fig.show()

def plot_correlations_plotly(df_articles, data_folder=None):
    """
    Regression and scatter plots (log-log) for clicks vs incoming links.
    Credits: https://community.plotly.com/t/adding-best-fit-line-linear-regression-line-in-a-scatter-plot/6069
    """
    
    fig_dumb, ax = plt.subplots()
    # Note: If your data is not already logged, regplot fits a linear line to linear data.
    # When displayed on log-axes in Plotly, this will look curved.
    rg = sns.regplot(data=df_articles, x="num_incoming_links", y="number_of_clicks", ax=ax, line_kws={"color": "red"})
    
    line = ax.lines[0] # Get the first line object
    X = line.get_xdata()
    Y = line.get_ydata()
    
    # Extract the Confidence Interval Path
    path_svg = ''
    if ax.collections:
        P = ax.collections[0].get_paths()
        p_codes = {1: 'M', 2: 'L', 79: 'Z'}
        
        # FIX: Unpack vertices and code directly
        for vertices, code in P[0].iter_segments():
            if code == 79: # CLOSEPOLY (Z)
                path_svg += "Z "
                continue
            
            c = p_codes.get(code, 'L')
            
            # FIX: Slice the last two elements [-2:] to get the (x, y) endpoint.
            # This handles lines (2 coords) AND curves (4 or 6 coords) safely.
            xx, yy = vertices[-2:] 
            
            path_svg += f"{c}{xx:.5f} {yy:.5f} "
            
    plt.close(fig_dumb) # Close the hidden plot

    # Create plotly interactive figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_articles["num_incoming_links"], y=df_articles["number_of_clicks"], 
        mode='markers', name='Data',
        marker=dict(opacity=0.5, size=8)
    ))

    # Regression line
    fig.add_trace(go.Scatter(
        x=X, y=Y, 
        mode='lines', name='Fit',
        line=dict(color='red')
    ))

    # CI
    if path_svg:
        fig.update_layout(shapes=[dict(
            type='path',
            path=path_svg,
            line=dict(width=0),
            fillcolor='rgba(68, 122, 219, 0.25)',
            xref='x', yref='y'
        )])

    # Apply Log Scales and Labels
    fig.update_layout(
        title="Pearson Correlation (with confidence interval)",
        xaxis_title="Number of Incoming Links",
        yaxis_title="Number of Clicks",
        xaxis_type="log",
        yaxis_type="log",
        template="plotly_white"
    )
    if data_folder:
        output_path = data_folder / "plot_correlations_plotly.html"
        fig.write_html(
            output_path, 
            include_plotlyjs='cdn'
        )
    fig.show()
    return fig

def plot_path_length_stats_plotly(df_lengths, smooth=False, resolution=500, max_n=None, data_folder=None):
    """
    Plot mean and median shortest-path lengths as a function of n.
    
    Parameters
    ----------
    df_lengths : DataFrame
        Output of path_length_stats(), must contain columns:
        ['n', 'mean_path_length', 'median_path_length']
    
    smooth : bool
        Whether to plot smooth interpolated curves.
    
    resolution : int
        Number of points for interpolation if smooth=True.
    """

    df = df_lengths.sort_values("n")
    
    ns = df["n"].values
    means = df["mean_path_length"].values
    medians = df["median_path_length"].values

    if not max_n is None:
        mask = ns <= max_n
        ns = ns[mask]
        means = means[mask]
        medians = medians[mask]

    # Plot
    df = pd.DataFrame({
        'n': ns,
        'Mean': means,
        'Median': medians
    })
    fig = px.line(df, x='n', y=['Mean', 'Median'], labels={'value': 'Shortest-path length'}, title="Mean and Median Shortest-Path Length vs n")
    fig.update_layout(hovermode="x")
    if data_folder:
        output_path = data_folder / "plot_path_length_stats_plotly.html"
        fig.write_html(
            output_path, 
            include_plotlyjs='cdn'
        )
    fig.show()
