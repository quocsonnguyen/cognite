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
                console.log(result);
                if (result.code === 0) {
                    // Render table
                    let tbody = $('#tbody')
                    tbody.empty()
                    for (let i = 0; i <= result.data.length; i++) {
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