function deleteRow(rowId) {
    $('#close-modal-btn').click()
    $.ajax({
        url: '/api/delete-record-in-global',
        type: 'post',
        data: {
            rowId : rowId
        }
    }).done(result => {
        if (result.code === 0) {
            $.ajax({
                url: '/api/load-global-table',
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
                                <td class="align-middle">${result.data[i][4]}</td>
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

$(document).ready(function () {
    $('#switch-btn').click(function () {
        let currentTable = $('#current-table').val()

        if (currentTable === 'global') {
            $.ajax({
                url: '/api/load-global-backup-table',
                type: 'get'
            }).done(result => {
                if (result.code === 0) {
                    let thead = $('#thead')
                    thead.html(`
                        <th scope="col" class="align-middle">#</th>
                        <th scope="col">Personalised Score</th>
                        <th scope="col">Current Intensity</th>
                        <th scope="col" class="align-middle">Performance</th>
                        <th scope="col" class="align-middle">Added Time</th>
                        <th scope="col" class="align-middle">User</th>
                    `)
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
                                <td class="align-middle">${result.data[i][4]}</td>
                            </tr>
                        `
                        tbody.append(tr)
                        $('#current-table').val('global_backup')
                        $('#type-of-data').text('Backup Data')
                        $('#switch-btn').text('Switch to Data file')
                    }
                } else {
                    $('#tbody').empty()
                }
            })
        }
        else if (currentTable === 'global_backup') {
            $.ajax({
                url: '/api/load-global-table',
                type: 'get'
            }).done(result => {
                if (result.code === 0) {
                    let thead = $('#thead')
                    thead.html(`
                        <th scope="col" class="align-middle">#</th>
                        <th scope="col">Personalised Score</th>
                        <th scope="col">Current Intensity</th>
                        <th scope="col" class="align-middle">Performance</th>
                        <th scope="col" class="align-middle">Added Time</th>
                        <th scope="col" class="align-middle">User</th>
                        <th scope="col" class="align-middle">Action</th>
                    `)
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
                                <td class="align-middle">${result.data[i][4]}</td>
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
                        $('#current-table').val('global')
                        $('#type-of-data').text('Data')
                        $('#switch-btn').text('Switch to Backup Data file')
                    }
                } else {
                    $('#tbody').empty()
                }
            })
        }
    })

    $('#v-btn').click(function () {
        let fromIndex = $('#from-i').val()
        let toIndex = $('#to-i').val()
        let skip = $('#skip').val()

        if (fromIndex && toIndex && skip && fromIndex > 0 && skip > 0) {
            if (toIndex > fromIndex || toIndex==-1) {
                $('#v-btn').addClass('d-none')
                $('#spinner').removeClass('d-none')
                $.ajax({
                    url: '/api/visualize',
                    type: 'get',
                    data : {
                        fromIndex : fromIndex,
                        toIndex : toIndex,
                        skip: skip
                    }
                }).done(result => {
                    $('#spinner').addClass('d-none')
                    $('#v-btn').removeClass('d-none')
                    if (result.code === 0) {
                        $('#current-intensity-inp').val(Number.parseFloat(result.data.currentIntensity).toFixed(4))
                        $('#p-score-inp').val(Number.parseFloat(result.data.personalisedScore).toFixed(4))
                        $('#noise-inp').val(result.data.noise)
                        $('#v-img-1').attr('src', '/api/image/gp_acq.png/' + Date.now())
                        $('#v-img-2').attr('src', '/api/image/gp_mean.png/' + Date.now())
                        $('#v-img-3').attr('src', '/api/image/gp_var.png/' + Date.now())
                    } else {
                        console.log('failed');
                    }
                })
            }
        }
    })
})