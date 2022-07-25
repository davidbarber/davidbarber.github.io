---
layout: default
permalink: index.html
title: David Barber
description: "Machine Learning"
---

## Machine Learning 


I'm fascinated about Artificial Intelligence and how to make computers smarter. We live in exciting times with rapid increases in data and compute resources. How can we use them to solve grand challenges in AI like

* Understanding text, images and video
* Trying to help us to reason about the world around us, like in digital assistants

I hope to post on this site some thoughts about machine learning.

---

* At UCL I lead a team of around 10 PhD students on machine learning with an emphasis on methodological developments. 

* I'm also Chief Scientific Officer at [reinfer.io](https://reinfer.io/) who develop next generation customer interaction and analytics engines.

Please also see [my UCL website]( http://www.cs.ucl.ac.uk/staff/d.barber) or you might also be looking for [my BRML textbook]( http://www.cs.ucl.ac.uk/staff/d.barber/brml). 

<!--
<a class="twitter-timeline" href="https://twitter.com/davidobarber">Tweets by davidobarber</a> <script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script>
-->

---

{% if site.twitter_widget_id %}
<div class="text-tweets">
<div class="tweets">
<a class="twitter-timeline"
  data-dnt="true"
  width="600"
  height="250"
  href="https://twitter.com/{{ site.owner.twitter }}"
  data-widget-id="{{ site.twitter_widget_id }}"
  data-tweet-limit="2"
  data-chrome="noheader nofooter noborders noscrollbar transparent">
  Recent Tweets</a>
</div>
<script>
    !function(d,s,id){var js,fjs=d.getElementsByTagName(s)[0],p=/^http:/.test(d.location)?'http':'https';if(!d.getElementById(id)){js=d.createElement(s);js.id=id;js.src=p+"://platform.twitter.com/widgets.js";fjs.parentNode.insertBefore(js,fjs);}}(document,"script","twitter-wjs");
</script>
</div>
{% else %}
{% endif %}


<h3 class="post-title">
<div class="pagination" style="margin: 0.5rem;">
    <a class="pagination-item older" href="{{ site.url }}/blog"><i class="fa fa-edit"> Blog</i></a>
    <a class="pagination-item newer" href="{{ site.url }}/tags"><i class="fa fa-tags"> Tags</i></a>
</div>
</h3>