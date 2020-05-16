<link rel="stylesheet" type="text/css" href="//cdn.jsdelivr.net/npm/slick-carousel@1.8.1/slick/slick.css"/>
<link rel="stylesheet" type="text/css" href="//cdn.jsdelivr.net/npm/slick-carousel@1.8.1/slick/slick-theme.css"/>
<link rel="stylesheet" type="text/css" href="style.css"/>

<script type="text/javascript" src="//code.jquery.com/jquery-1.11.0.min.js"></script>
<script type="text/javascript" src="//code.jquery.com/jquery-migrate-1.2.1.min.js"></script>
<script type="text/javascript" src="//cdn.jsdelivr.net/npm/slick-carousel@1.8.1/slick/slick.min.js"></script>

# Author
Quei-An Chen (kwea123). Photo credits: Ben Mildenhall ([bmild](https://github.com/bmild))

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

# Real time volume rendering in Unity
[Volume rendering](https://en.wikipedia.org/wiki/Volume_rendering) is a technique that doesn't require "real object". The model you see here is composed of rays, so we can cut off parts to see internal structures, also perform deforming effect in real time.
<iframe src="https://i.simmer.io/@kwea123/nerf-volume-rendering" style="width:960px;height:600px;"></iframe>

<br/>

# Colored mesh reconstruction (photogrammetry)
We can also generate real colored mesh that allows the object to interact with other physical objects.
<iframe src="https://i.simmer.io/@kwea123/nerf-mesh" style="width:960px;height:600px;"></iframe>

<br/>

# Mixed reality in Unity (to be updated)

<br/>

# Tutorial

I also have tutorials on how to achieve above results using google colab:

<a href="https://www.youtube.com/playlist?list=PLDV2CyUo4q-K02pNEyDr7DYpTQuka3mbV">
<img src="https://user-images.githubusercontent.com/11364490/80913471-d5781080-8d7f-11ea-9f72-9d68402b8271.png" style="display:block;margin:auto;">
</a>