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
                        $('#switch-btn').text('Switch to GLOBAL file')
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
                        $('#switch-btn').text('Switch to GLOBAL BACKUP file')
                    }
                } else {
                    $('#tbody').empty()
                }
            })
        }
    })
})