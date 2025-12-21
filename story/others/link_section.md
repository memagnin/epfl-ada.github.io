---
layout: page
title: Link sections
subtitle: "Fun Facts"
---

To look at where are distributed the links we could make a [Venn diagram](https://en.wikipedia.org/wiki/Venn_diagram), but scale it would not be the fanciest.
Instead lets do an upset plot:

{% include upsetplot_section_category.html %}

This plot is like a Vienn plot. We can see that the body has a lot of unique links (1086) whereas, on the contrary, the infobox has only a few (64). Most links are both in the [lead]({{ '/story/1_b' | relative_url }}) and the body and many are in all three section. That means that if we look at the links in the [lead]({{ '/story/1_b' | relative_url }}) only, we won't miss a lot of links as most of the links in the body are already in the [lead]({{ '/story/1_b' | relative_url }}).

This plot still do not look at the [players behaviour]({{ '/story/2_a' | relative_url }}).

