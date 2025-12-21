---
layout: page
title: HITS score
subtitle: "Fun Facts"
---

From [Wikipedia](https://en.wikipedia.org/wiki/HITS_algorithm), the free encyclopedia

Hyperlink-Induced Topic Search (HITS; also known as hubs and authorities) is a link analysis algorithm that rates Web pages, developed by Jon Kleinberg. The idea behind Hubs and Authorities stemmed from a particular insight into the creation of web pages when the Internet was originally forming; that is, certain web pages, known as hubs, served as large directories that were not actually authoritative in the information that they held, but were used as compilations of a broad catalog of information that led users direct to other authoritative pages. In other words, a good hub represents a page that pointed to many other pages, while a good authority represents a page that is linked by many different hubs.[1](https://nlp.stanford.edu/IR-book/html/htmledition/hubs-and-authorities-1.html)

It has not been developped directly for the [structure of Wikipedia]({{ '/story/1_a' | relative_url }}) nor does it take into account [where the link is placed]({{ '/story/1_b' | relative_url }}). Wikipedia would be just a [graph]({{ '/story/1_a' | relative_url }})

<a title="Андронов Руслан, Public domain, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:SetsEN.jpg"><img width="512" alt="SetsEN" src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/SetsEN.jpg/512px-SetsEN.jpg?20130115030318"></a>

#### Pseudo code:
Here is te pseucode of the HITS algorithm
```
G := set of pages
for each page p in G do
    p.auth = 1 // p.auth is the authority score of the page p
    p.hub = 1 // p.hub is the hub score of the page p
for step from 1 to k do // run the algorithm for k steps
    norm = 0
    for each page p in G do  // update all authority values first
        p.auth = 0
        for each page q in p.incomingNeighbors do // p.incomingNeighbors is the set of pages that link to p
            p.auth += q.hub
        norm += square(p.auth) // calculate the sum of the squared auth values to normalise
    norm = sqrt(norm)
    for each page p in G do  // update the auth scores 
        p.auth = p.auth / norm  // normalise the auth values
    norm = 0
    for each page p in G do  // then update all hub values
        p.hub = 0
        for each page r in p.outgoingNeighbors do // p.outgoingNeighbors is the set of pages that p links to
            p.hub += r.auth
        norm += square(p.hub) // calculate the sum of the squared hub values to normalise
    norm = sqrt(norm)
    for each page p in G do  // then update all hub values
        p.hub = p.hub / norm   // normalise the hub values
```

more details can be found [here](https://nlp.stanford.edu/IR-book/html/htmledition/hubs-and-authorities-1.html)

#### Pagerank
Careful even if the HITS score has similar properties with [PageRank]({{ '/story/others/PageRank' | relative_url }}), they are **not** the same.

#### Licence note:
**Attribution:** This page includes material adapted from the Wikipedia article
[“HITS algorithm”](https://en.wikipedia.org/wiki/HITS_algorithm)
(see [page history](https://en.wikipedia.org/w/index.php?title=HITS_algorithm&action=history)),
used under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.
Changes: formatting and edits; added pseudocode and extra notes.