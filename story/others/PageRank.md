---
layout: page
title: PageRank
subtitle: "Fun Facts"
---

#### Description
PageRank is a link analysis algorithm and it assigns a numerical weighting to each element of a hyperlinked set of documents, such as the World Wide Web, with the purpose of "measuring" its relative importance within the set. The algorithm may be applied to any collection of entities with reciprocal quotations and references. The numerical weight that it assigns to any given element E is referred to as the PageRank of E and denoted by P R ( E ) . {\displaystyle PR(E).}

A PageRank results from a mathematical algorithm based on the Webgraph, created by all World Wide Web pages as nodes and [hyperlinks as edges]({{ '/story/1_a' | relative_url }}), taking into consideration authority hubs such as cnn.com or mayoclinic.org. The rank value indicates an importance of a particular page. A hyperlink to a page counts as a vote of support. The PageRank of a page is defined recursively and depends on the number and PageRank metric of all pages that link to it ("incoming links"). A page that is linked to by many pages with high PageRank receives a high rank itself.[1](https://www.tandfonline.com/doi/citedby/10.1080/15326340600649052?scroll=top&needAccess=true)

Numerous academic papers concerning PageRank have been published since Page and Brin's original paper.[2](http://infolab.stanford.edu/pub/papers/google.pdf) In practice, the PageRank concept may be vulnerable to manipulation. Research has been conducted into identifying falsely influenced PageRank rankings. The goal is to find an effective means of ignoring links from documents with falsely influenced PageRank.[3](http://ilpubs.stanford.edu:8090/697/1/2005-33.pdf)

Other link-based ranking algorithms for Web pages include the HITS algorithm invented by Jon Kleinberg (used by Teoma and now Ask.com), the IBM CLEVER project, the TrustRank algorithm, the Hummingbird algorithm,[4](https://searchengineland.com/google-hummingbird-172816) and the SALSA algorithm.

<a title="Sage santo, CC BY-SA 4.0 &lt;https://creativecommons.org/licenses/by-sa/4.0&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Page_rank_animation.gif"><img width="512" alt="Page rank animation" src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Page_rank_animation.gif/512px-Page_rank_animation.gif?20250414161624"></a>

Of course, tt has not been developped directly for the [structure of Wikipedia]({{ '/story/1_a' | relative_url }}) nor does it take into account [where the link is placed]({{ '/story/1_b' | relative_url }}). Wikipedia would be just a [graph]({{ '/story/1_a' | relative_url }})

#### HITS
Careful even if the HITS score has similar properties with [PageRank]({{ '/story/others/HITS' | relative_url }}), they are **not** the same.

#### Licence note:
**Attribution:** This page includes material adapted from the Wikipedia article
[“PageRank”](hhttps://en.wikipedia.org/wiki/PageRank)
(see [page history](https://en.wikipedia.org/w/index.php?title=PageRank&action=history)),
used under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.
Changes: formatting and edits; added pseudocode and extra notes.