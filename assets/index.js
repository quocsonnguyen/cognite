function deleteRow(rowId) {
    $('#close-modal-btn').click()
    $.ajax({
        url: '/api/delete-record',
        type: 'post',
        data: {
            rowId : rowId
        }
    }).done(result => {
        if (result.code === 0) {
            $.ajax({
                url: '/api/load-table',
                type: 'get'
            }).done(result => {
                if (result.code === 0) {
                    // Render table
                    let tbody = $('#tbody')
                    tbody.empty()
                    for (let i = 0; i <= result.data.length; i++) {
                        if (result.data[i] === undefined) { continue }
                        let tr = document.createElement('tr')
                        tr.innerHTML = `
                            <tr>
                                <th class="align-middle"  scope="row">${i+1}</th>
                                <td class="align-middle">${result.data[i][1]}</td>
                                <td class="align-middle">${result.data[i][0]}</td>
                                <td class="align-middle">${result.data[i][2]}</td>
                                <td class="align-middle">${result.data[i][3]}</td>
                                <td class="align-middle">
                                    <button 
                                        class="button-delete" type="button" 
                                        data-toggle="modal" data-target="#confirmDeleteModal"
                                        onclick="preDeleteRow(${i+1})">
                                        Delete
                                    </button>
                                </td>
                            </tr>
                        `
                        tbody.append(tr)
                    }
                } else {
                    $('#tbody').empty()
                }
            })
        }
    })
}

function preDeleteRow(rowId) {
    $('#confirm-delete-btn').removeAttr('onclick');
    $('#confirm-delete-btn').attr('onClick', `deleteRow(${rowId})`);
}

$(document).ready(function(){
    $('#get-freq-btn').click(() => {
        let ps = $('#personalised-score').val()
        ps = Number.parseFloat(ps)

        if (ps > 0) {
            $.ajax({
                url: '/api/get-freq',
                type: 'get',
                data: {
                    ps: ps
                }
            }).done(function(result) {
                console.log(result);
                $('#freq').val(result.freq.toFixed(2))
            })
        } else {
            $('#alert-msg').removeClass('d-none')   
            setTimeout(() => {
                $('#alert-msg').addClass('d-none')   
            }, 4000)
        }
    })

    $('#save-record-btn').click(() => {
        let freq = $('#freq').val()
        let ps = $('#personalised-score').val()
        let p = $('#performance').val()

        freq = Number.parseFloat(freq)
        ps = Number.parseFloat(ps)
        p = Number.parseFloat(p)

        if (p <= 0) {
            $('#alert-msg').removeClass('d-none')   
            setTimeout(() => {
                $('#alert-msg').addClass('d-none')   
            }, 4000)

            return
        }

        if (freq && ps && p) {
            $.ajax({
                url: '/api/save-new-record',
                type: 'post',
                dataType: 'json',
                data: {
                    freq : freq,
                    ps : ps,
                    p : p
                }
            }).done(function(result) {
                console.log(result);
                if (result.code === 0) {
                    let count = $('#tbody tr').length + 1
                    let tr = document.createElement('tr')
                    tr.innerHTML = `
                        <tr>
                            <th class="align-middle" scope="row">${count}</th>
                            <td class="align-middle">${ps.toFixed(2)}</td>
                            <td class="align-middle">${freq.toFixed(1)}</td>
                            <td class="align-middle">${p.toFixed(1)}</td>
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