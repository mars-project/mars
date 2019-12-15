$(function() {
    var resize_page = function() {
        var page_wrapper_obj = $('#page-wrapper');
        var new_height = $(window).height() - $('.navbar-static-top').outerHeight() - $('#page-nav').outerHeight();
        if (new_height >= page_wrapper_obj[0].scrollHeight)
            page_wrapper_obj.css('height', new_height);
        else
            page_wrapper_obj.css('height', '');
        window.setTimeout(resize_page, 1000);
    };
    resize_page();
    $(window).resize(resize_page);

    // put pill tags into url hash to facilitate page reloading
    $('a[data-toggle="pill"]').click(function () {
        window.location.hash = $(this).attr('href');
    });
    if (window.location.hash) {
        $('a[href=\'' + window.location.hash + '\']').click();
    }

    // reload auto-refresh tables while preserving orders
    var refresh_tables = function () {
        var refreshs = $('.auto-refresh').map(function () {
            var that = this;
            var retInfo = { id: $(this).attr('id') };
            $(that).find('th').each(function () {
                if ($(this).find('.asc').length > 0 || $(this).find('.desc').length > 0)
                    retInfo = { id: retInfo.id, sortName: $(this).data('field'),
                        sortOrder: $(this).data('order') };
            });
            return retInfo;
        });
        $.get(document.URL, function(txt) {
            var newDom = $(txt);
            $.each(refreshs, function (i, refInfo) {
                var tableDivObj = newDom.find('#' + refInfo.id);
                if (refInfo.sortName !== undefined) {
                    tableDivObj.find('table')
                        .attr('data-sort-name', refInfo.sortName)
                        .attr('data-sort-order', refInfo.sortOrder);
                }
                $('#' + refInfo.id).replaceWith(tableDivObj);
                $('#' + refInfo.id + ' table').bootstrapTable();
            });
        }, 'text');
        window.setTimeout(refresh_tables, 5000);
    };
    if ($('.auto-refresh').length > 0) {
        window.setTimeout(refresh_tables, 5000);
    }
});
