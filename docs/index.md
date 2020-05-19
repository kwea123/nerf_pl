<link rel="stylesheet" type="text/css" href="//cdn.jsdelivr.net/npm/slick-carousel@1.8.1/slick/slick.css"/>
<link rel="stylesheet" type="text/css" href="//cdn.jsdelivr.net/npm/slick-carousel@1.8.1/slick/slick-theme.css"/>
<link rel="stylesheet" type="text/css" href="style.css"/>

<script type="text/javascript" src="//code.jquery.com/jquery-1.11.0.min.js"></script>
<script type="text/javascript" src="//code.jquery.com/jquery-migrate-1.2.1.min.js"></script>
<script type="text/javascript" src="//cdn.jsdelivr.net/npm/slick-carousel@1.8.1/slick/slick.min.js"></script>
<script async defer src="https://buttons.github.io/buttons.js"></script>

# Author of this implementation
<h3>If you find my work helpful, please consider star this project!
  <a class="github-button" href="https://github.com/kwea123/nerf_pl" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star kwea123/nerf_pl on GitHub">Star</a>
</h3>

Quei-An Chen ([kwea123](https://github.com/kwea123)). Original author and photo credits: Ben Mildenhall ([bmild](https://github.com/bmild))

# What can NeRF do?
<img src="https://user-images.githubusercontent.com/11364490/82124460-1ccbbb80-97da-11ea-88ad-25e22868a5c1.png" style="max-width:100%">

<br/>

# 360 degree view synthesis
<div class="nerf_mp4">
  <video autoplay controls loop>
    <source src="https://storage.cloud.google.com/kwea123_dataset/nerf/pond.mp4" type="video/mp4">
  </video>
  <video autoplay controls loop>
    <source src="https://storage.cloud.google.com/kwea123_dataset/nerf/trex.mp4" type="video/mp4">
  </video>
  <video autoplay controls loop>
    <source src="https://storage.cloud.google.com/kwea123_dataset/nerf/horns.mp4" type="video/mp4">
  </video>
  <video autoplay controls loop>
    <source src="https://storage.cloud.google.com/kwea123_dataset/nerf/silica2.mp4" type="video/mp4">
  </video>
  <video autoplay controls loop>
    <source src="https://storage.cloud.google.com/kwea123_dataset/nerf/duorou.mp4" type="video/mp4">
  </video>
</div>

<script>
$(document).ready(function(){
  $('.nerf_mp4').slick({
    slidesToShow: 3,
    slidesToScroll: 1,
    dots: true,
    autoplay: true,
    autoplaySpeed: 3000,
    infinite: true,
  });
});
</script>

<br/>

# Colored 3D mesh reconstruction (photogrammetry)
We can generate real colored mesh that allows the object to interact with other physical objects.
<iframe src="https://i.simmer.io/@kwea123/nerf-mesh" style="width:960px;height:600px;"></iframe>

<br/>

# Real time volume rendering in Unity
[Volume rendering](https://en.wikipedia.org/wiki/Volume_rendering) is a technique that doesn't require "real object". The model you see here is composed of rays, so we can cut off parts to see internal structures, also perform deforming effect in real time.
<iframe src="https://i.simmer.io/@kwea123/nerf-volume-rendering" style="width:960px;height:600px;"></iframe>

<br/>

# Mixed reality in Unity (doesn't work on FireFox, please use Chrome)
Accurate depth allows us to embed virtual object inside real scenes with correct z-order.

<iframe src="https://i.simmer.io/@kwea123/nerf-mixed-reality" style="width:960px;height:600px;"></iframe>
<br/>

# Tutorial

I also have tutorials on how to achieve above results using google colab:

<a href="https://www.youtube.com/playlist?list=PLDV2CyUo4q-K02pNEyDr7DYpTQuka3mbV">
<img src="https://user-images.githubusercontent.com/11364490/80913471-d5781080-8d7f-11ea-9f72-9d68402b8271.png" style="display:block;margin:auto;">
</a>

# Call for contribution
If you are expert in Unity and know how to make more visual appealing effects for the models shown above, feel free to contact me! I can share my code and data with you, and put your name on my github page!
