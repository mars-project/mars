$(function() {
    var resize_page = function() {
        var new_height = $(window).height() - $('.navbar-static-top').outerHeight();
        var page_wrapper_obj = $('#page-wrapper');
        if (new_height >= page_wrapper_obj[0].scrollHeight)
            page_wrapper_obj.css('height', new_height);
        else
            page_wrapper_obj.css('height', '');
        window.setTimeout(resize_page, 1000);
    };
    resize_page();
    $(window).resize(resize_page);
});