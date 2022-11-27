function deleteRow(rowId) {
    console.log(rowId);
    $('#close-modal-btn').click()
    $.ajax({
        url: '/api/delete-record',
        type: 'post',
        data: {
            rowId : rowId
        }
    }).done(result => {
        if (result.code === 0) {
            location.reload()
        }
    })
}

function preDeleteRow(rowId) {
    $('#confirm-delete-btn').removeAttr('onclick');
    $('#confirm-delete-btn').attr('onClick', `deleteRow(${rowId})`);
}

function editRow(sh, bs, si) {
    $('#spherical-head').val(sh)
    $('#behavioral-score').val(bs)
    $('#freq').val(si)

    console.log(this);

}

$(document).ready(function(){
    $('#get-freq-btn').click(() => {
        let sh = $('#spherical-head').val()
        sh = Number.parseFloat(sh)
        
        // Validate for spherical-head (48-70)
        if (!(sh >= 48 && sh <= 70)) {
            $('#alert-msg').html('Spherical Head must be in range [48-70]')
            $('#alert-msg').removeClass('d-none')   
            setTimeout(() => {
                $('#alert-msg').addClass('d-none')   
            }, 4000)
            return;
        }

        let bs = $('#behavioral-score').val()
        bs = Number.parseFloat(bs)

        // Validate for behavioral-score (0-1)
        if (!(bs >= 0 && bs <= 1)) {
            $('#alert-msg').html('Behavioral Score must be in range [0-1]')
            $('#alert-msg').removeClass('d-none')   
            setTimeout(() => {
                $('#alert-msg').addClass('d-none')   
            }, 4000)
            return; 
        }

        $.ajax({
            url: '/api/get-freq',
            type: 'get',
            data: {
                sh: sh,
                bs: bs,
            }
        }).done(function(result) {
            $('#freq').val(result.freq.toFixed(1))

            let sh = $('#spherical-head').val()
            let bs = $('#behavioral-score').val()

            let tbody = $('#tbody')

            console.log(tbody);

            let tr = document.createElement('tr')
            tr.innerHTML = `
                <tr>
                    <th class="align-middle"  scope="row">${tbody[0].children.length + 1}</th>
                    <td class="align-middle">${sh}</td>
                    <td class="align-middle">${bs}</td>
                    <td class="align-middle">${result.freq.toFixed(1)}</td>
                    <td class="align-middle">-</td>
                    <td class="align-middle">-</td>
                    <td class="align-middle">${result.time}</td>
                    <td class="align-middle">
                        <button
                            class="button-edit" type="button"
                            onclick="editRow(${sh}, ${bs}, ${result.freq.toFixed(1)})">
                            Edit
                        </button>
                         |  
                        <button 
                            class="button-delete" type="button" 
                            data-toggle="modal" data-target="#confirmDeleteModal"
                            onclick="preDeleteRow(${tbody[0].children.length + 1})">
                            Delete
                        </button>
                    </td>
                </tr>
            `
            tbody.append(tr)
        })
    })

    $('#save-record-btn').click(() => {
        let sh = $('#spherical-head').val()
        let bs = $('#behavioral-score').val()
        let freq = $('#freq').val()
        let o = $('#outcome').val()
        
        sh = Number.parseFloat(sh)
        bs = Number.parseFloat(bs)
        freq = Number.parseFloat(freq)
        o = Number.parseFloat(o)

        if (o <= 0) {
            $('#alert-msg').html('Outcome must be greater than 0')
            $('#alert-msg').removeClass('d-none')   
            setTimeout(() => {
                $('#alert-msg').addClass('d-none')   
            }, 4000)

            return
        }

        if (sh && bs && freq && o) {
            $.ajax({
                url: '/api/save-new-record',
                type: 'post',
                dataType: 'json',
                data: {
                    sh : sh,
                    bs : bs,
                    freq: freq,
                    o : o,
                }
            }).done(function(result) {
                if (result.code === 0) {
                    let ratio = Number.parseFloat(o) / Number.parseFloat(bs)
                    let count = $('#tbody tr').length + 1
                    let tr = document.createElement('tr')
                    tr.innerHTML = `
                        <tr>
                            <th class="align-middle" scope="row">${count}</th>
                            <td class="align-middle">${sh}</td>
                            <td class="align-middle">${bs.toFixed(4)}</td>
                            <td class="align-middle">${freq.toFixed(1)}</td>
                            <td class="align-middle">${o.toFixed(4)}</td>
                            <td class="align-middle">${ratio.toFixed(4)}</td>
                            <td class="align-middle">${result.addedTime}</td>
                            <td class="align-middle">
                                <button 
                                    class="button-delete" type="button" 
                                    data-toggle="modal" data-target="#confirmDeleteModal"
                                    onclick="preDeleteRow(${count})">
                                    Delete
                                </button>
                            </td>
                        </tr>
                    `
                    $('#tbody').append(tr)
                }
            })
        }
    })
});