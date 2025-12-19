---
layout: page
title: Different Distribution
---

In this plot:
{% include plot_stacked_path_length_distribution_plotly.html %}

We look at the distribution as an histogram (through the stacked bins). But to confirm there is actually a trend, it is better to look at some statistical metric. We could use [box plots](https://en.wikipedia.org/wiki/Box_plot) but it would not be pretty to see the *evolution*. We can just plot the mean and median

{% include plot_path_length_stats_plotly.html %}


We observe exactly what we expected. At the [beginning]({{ '/story/1_b' | relative_url }}), the shortest paths length increases because more paths are added without shortcuts and as `n` increases, we don't add a lot of path but we add connectivity which reduces the shortest path lengths.

We could do the same plot but seeing how many [players paths]({{ '/story/2_a' | relative_url }}) would be still possible. Even further, we could see if it is related to a player finishing or not. If he has to scroll a lot to finish a lot, he might abandon earlier.