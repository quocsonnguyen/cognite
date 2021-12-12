function getTime() {
    let today  = new Date();
    let dd = String(today.getDate()).padStart(2, '0');
    let mm = String(today.getMonth() + 1).padStart(2, '0'); //January is 0!
    let yyyy = today.getFullYear();
    let hour = String(today.getHours()).padStart(2, '0');
    let minute = String(today.getMinutes()).padStart(2, '0')
    let createdTime = dd + '/' + mm + '/' + yyyy + ' - ' + hour + ':' + minute;
    return createdTime
}

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
                console.log(result);
                if (result.code === 0) {
                    // Render table
                    let tbody = $('#tbody')
                    tbody.empty()
                    for (let i = 0; i <= result.data.length; i++) {
                        let tr = document.createElement('tr')
                        tr.innerHTML = `
                            <tr>
                                <th scope="row">${i+1}</th>
                                <td>${result.data[i][0]}</td>
                                <td>${result.data[i][1]}</td>
                                <td>${result.data[i][2]}</td>
                                <td>${result.data[i][3]}</td>
                                <td>
                                    <button 
                                        class="btn btn-danger type="button" 
                                        data-toggle="modal" data-target="#confirmDeleteModal"
                                        onclick="preDeleteRow(${i+1})">
                                        Delete
                                    </button>
                                </td>
                            </tr>
                        `
                        tbody.append(tr)
                    }
                }
            })
        }
    })
}

function preDeleteRow(rowId) {
    console.log(rowId);
    $('#confirm-delete-btn').removeAttr('onclick');
    $('#confirm-delete-btn').attr('onClick', `deleteRow(${rowId})`);
}

$(document).ready(function(){
    $('#get-freq-btn').click(() => {
        let ps = $('#personalised-score').val()
        ps = Number.parseFloat(ps)
        if (ps) {
            $.ajax({
                url: '/api/get-freq',
                type: 'get',
                data: {
                    ps: ps
                }
            }).done(function(result) {
                $('#freq').val(result.freq.toFixed(2))
                console.log(result);
            })
        }
    })

    $('#save-record-btn').click(() => {
        let freq = $('#freq').val()
        let ps = $('#personalised-score').val()
        let p = $('#performance').val()

        freq = Number.parseFloat(freq)
        ps = Number.parseFloat(ps)
        p = Number.parseFloat(p)
        addedTime = getTime()

        if (freq && ps && p) {
            $.ajax({
                url: '/api/save-new-record',
                type: 'post',
                dataType: 'json',
                data: {
                    freq : freq,
                    ps : ps,
                    p : p,
                    addedTime : addedTime
                }
            }).done(function(result) {
                console.log(result);
                if (result.code === 0) {
                    let count = $('#tbody tr').length + 1
                    let tr = document.createElement('tr')
                    tr.innerHTML = `
                        <tr>
                            <th scope="row">${count}</th>
                            <td>${freq.toFixed(1)}</td>
                            <td>${ps.toFixed(2)}</td>
                            <td>${p.toFixed(1)}</td>
                            <td>${addedTime}</td>
                            <td>
                                <button 
                                    class="btn btn-danger type="button" 
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