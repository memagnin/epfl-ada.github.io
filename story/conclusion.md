---
layout: page
title: "Conclusion"
mathjax: true
---

to include a static image (add in the img folder): ![crepe]({{ '/assets/img/crepe.jpg' | relative_url }})

to include an interactive plot (add the html in the include folder): 
{% include links_sections.html %}

to add hyperlinks: [finish]({{ '/story/others/fun_fact' | relative_url }})

to latex double dollars always $$\alpha$$


Here is the graph of the game! 

![game graph]({{ '/assets/img/gamegraph.png' | relative_url }})


And if you want to look at the wrong links (Not ordered):
- [BERT]({{ '/story/others/BERT' | relative_url }})
- [The model]({{ '/story/others/BERT' | relative_url }})
- [A fun fact]({{ '/story/others/fun_fact' | relative_url }})
- [HITS]({{ '/story/others/HITS' | relative_url }})
- [A huge set]({{ '/story/others/huge_set' | relative_url }})
- [An upset plot]({{ '/story/others/link_section' | relative_url }})
- [Mean and Median]({{ '/story/others/mean_median' | relative_url }})
- [PageRank]({{ '/story/others/PageRank' | relative_url }})
- [Wikipedia]({{ '/story/others/Wikipedia' | relative_url }})
- More links to other websites were included like papers, wikipedia pages etc.