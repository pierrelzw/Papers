function spider_frontend_ajax(form_id, current_view, id, album_gallery_id, cur_album_id, type, srch_btn, title) {
  var page_number = jQuery("#page_number_" + current_view).val();
  var bwg_previous_album_ids = jQuery('#bwg_previous_album_id_' + current_view).val();
  var bwg_previous_album_page_numbers = jQuery('#bwg_previous_album_page_number_' + current_view).val();
  var post_data = {};
  if (album_gallery_id == 'back') { // Back from album.
    var bwg_previous_album_id = bwg_previous_album_ids.split(",");
    album_gallery_id = bwg_previous_album_id[0];
    jQuery('#bwg_previous_album_id_' + current_view).val(bwg_previous_album_ids.replace(bwg_previous_album_id[0] + ',', ''));
    var bwg_previous_album_page_number = bwg_previous_album_page_numbers.split(",");
    page_number = bwg_previous_album_page_number[0];
    jQuery('#bwg_previous_album_page_number_' + current_view).val(bwg_previous_album_page_numbers.replace(bwg_previous_album_page_number[0] + ',', ''));
  }
  else if (cur_album_id != '') { // Enter album (not change the page).
    jQuery('#bwg_previous_album_id_' + current_view).val(cur_album_id + ',' + bwg_previous_album_ids);
    if (page_number) {
      jQuery('#bwg_previous_album_page_number_' + current_view).val(page_number + ',' + bwg_previous_album_page_numbers);
    }
    page_number = 1;
  }
  if (srch_btn) { // Start search.
    page_number = 1; 
  }
  if (typeof title == "undefined") {
    var title = "";
  }
  post_data["page_number_" + current_view] = page_number;
  post_data["album_gallery_id_" + current_view] = album_gallery_id;
  post_data["bwg_previous_album_id_" + current_view] = jQuery('#bwg_previous_album_id_' + current_view).val();
  post_data["bwg_previous_album_page_number_" + current_view] = jQuery('#bwg_previous_album_page_number_' + current_view).val();
  post_data["type_" + current_view] = type;
  post_data["title_" + current_view] = title;
  if (jQuery("#bwg_search_input_" + current_view).length > 0) { // Search box exists.
    post_data["bwg_search_" + current_view] = jQuery("#bwg_search_input_" + current_view).val();
  }
  // Loading.
  jQuery("#ajax_loading_" + current_view).css('display', '');
  jQuery.post(
    window.location,
    post_data,
    function (data) {
      var str = jQuery(data).find('#' + form_id).html();
      jQuery('#' + form_id).html(str);
      // There are no images.
      if (jQuery("#bwg_search_input_" + current_view).length > 0 && album_gallery_id == 0) { // Search box exists and not album view.
        var bwg_images_count = jQuery('#bwg_images_count_' + current_view).val();
        if (bwg_images_count == 0) {
          var cont = jQuery("#" + id).parent().html();
          var error_msg = '<div style="width:95%"><div class="error"><p><strong>' + bwg_objectL10n.bwg_search_result + '</strong></p></div></div>';
          jQuery("#" + id).parent().html(error_msg + cont)
        }
      }
    }
  ).success(function(jqXHR, textStatus, errorThrown) {
    jQuery("#ajax_loading_" + current_view).css('display', 'none');
    // window.scroll(0, spider_get_pos(document.getElementById(form_id)));
    jQuery("html, body").animate({scrollTop: jQuery('#' + form_id).offset().top - 150}, 500);
    // For masonry view.
    var cccount = 0;
    var obshicccount = jQuery(".bwg_masonry_thumb_spun_" + current_view + " a").length;
    jQuery(".bwg_masonry_thumb_spun_" + current_view + " a img").on("load", function() {
      if (++cccount == obshicccount) {
        window["bwg_masonry_" + current_view]();
      }
    });
  });
  // if (event.preventDefault) {
    // event.preventDefault();
  // }
  // else {
    // event.returnValue = false;
  // }
  return false;
}
